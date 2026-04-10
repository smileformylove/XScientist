import argparse
import json
import os
import os.path as osp
import re
import shutil
import subprocess
import traceback
import unicodedata
import uuid

from ai_scientist.llm import (
    get_response_from_llm,
    extract_json_between_markers,
    create_client,
    AVAILABLE_LLMS,
)

from ai_scientist.tools.semantic_scholar import search_for_papers
from ai_scientist.utils.pipeline_helpers import (
    compile_latex as shared_compile_latex,
    iter_bfts_run_dirs,
)
from ai_scientist.utils.latex_lint import run_chktex
from ai_scientist.utils.manuscript_state import render_manuscript_prompt_context
from ai_scientist.utils.pipeline_contracts import load_contract_artifact
from ai_scientist.writing_prompt_profiles import (
    DEFAULT_WRITING_PROFILE,
    list_writing_profiles,
    normalize_writing_profile,
    render_writing_profile_self_checks,
    render_writing_profile_system_guidance,
)
from ai_scientist.writing_skill_pack import render_writing_skill_pack
from ai_scientist.writing_audit import (
    format_writing_audit_for_prompt,
    run_writing_audit,
)
from ai_scientist.writeup_guardrails import (
    build_submission_guardrail_report,
    build_citation_consistency_report,
    build_guardrail_repair_plan,
    collect_guardrail_findings,
    has_blocking_guardrail_violations,
    list_blocking_guardrail_reasons,
    render_citation_integrity_rules,
    render_humanizer_style_rules,
    render_venue_checklist,
)

from ai_scientist.perform_vlm_review import generate_vlm_img_review
from ai_scientist.vlm import create_client as create_vlm_client
from ai_scientist.utils.auth_session import require_login


def remove_accents_and_clean(s):
    # print("Original:", s)
    # Normalize to separate accents
    nfkd_form = unicodedata.normalize("NFKD", s)
    # Remove non-ASCII characters
    ascii_str = nfkd_form.encode("ASCII", "ignore").decode("ascii")
    # Remove anything but letters, digits, underscores, colons, dashes, @, {, }, and now commas
    ascii_str = re.sub(r"[^a-zA-Z0-9:_@\{\},-]+", "", ascii_str)
    # Convert to lowercase
    ascii_str = ascii_str.lower()
    # print("Cleaned: ", ascii_str)
    return ascii_str


def compile_latex(cwd, pdf_file, timeout=30):
    return shared_compile_latex(cwd, pdf_file, timeout=timeout)


def _resolve_writing_profile(profile: str | None) -> str:
    requested = profile or os.environ.get(
        "AI_SCIENTIST_WRITING_PROFILE", DEFAULT_WRITING_PROFILE
    )
    try:
        return normalize_writing_profile(requested)
    except ValueError as exc:
        print(f"Warning: {exc}; falling back to '{DEFAULT_WRITING_PROFILE}'.")
        return DEFAULT_WRITING_PROFILE


def detect_pages_before_impact(latex_folder, timeout=30):
    """
    Temporarily copy the latex folder, compile, and detect on which page
    the phrase "Impact Statement" appears.
    Returns a tuple (page_number, line_number) if found, otherwise None.
    """
    temp_dir = osp.join(latex_folder, f"_temp_compile_{uuid.uuid4().hex}")
    try:
        shutil.copytree(latex_folder, temp_dir, dirs_exist_ok=True)

        # Compile in the temp folder
        commands = [
            ["pdflatex", "-interaction=nonstopmode", "template.tex"],
            ["bibtex", "template"],
            ["pdflatex", "-interaction=nonstopmode", "template.tex"],
            ["pdflatex", "-interaction=nonstopmode", "template.tex"],
        ]
        for command in commands:
            try:
                subprocess.run(
                    command,
                    cwd=temp_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                return None

        temp_pdf_file = osp.join(temp_dir, "template.pdf")
        if not osp.exists(temp_pdf_file):
            return None

        # Try page-by-page extraction to detect "Impact Statement"
        for i in range(1, 51):
            page_txt = osp.join(temp_dir, f"page_{i}.txt")
            subprocess.run(
                [
                    "pdftotext",
                    "-f",
                    str(i),
                    "-l",
                    str(i),
                    "-q",
                    temp_pdf_file,
                    page_txt,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if not osp.exists(page_txt):
                break
            with open(page_txt, "r", encoding="utf-8", errors="ignore") as fp:
                page_content = fp.read()
            lines = page_content.split("\n")
            for idx, line in enumerate(lines):
                if "Impact Statement" in line:
                    return (i, idx + 1)
        return None
    except Exception:
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_citation_addition(
    client, model, context, current_round, total_rounds, idea_text
):
    report, citations = context
    msg_history = []
    citation_system_msg_template = """You are an ambitious AI researcher who is looking to publish a paper to a top-tier ML conference that will contribute significantly to the field.
You have already completed the experiments and now you are looking to collect citations to related papers.
This phase focuses on collecting references and annotating them to be integrated later.
Collected citations will be added to a references.bib file.

Reasons to reference papers include:
1. Summarizing Research: Cite sources when summarizing the existing literature.
2. Using Specific Concepts or Data: Provide citations when discussing specific theories, models, or data.
3. Comparing Findings: Cite relevant studies when comparing or contrasting different findings.
4. Highlighting Research Gaps: Cite previous research when pointing out gaps your survey addresses.
5. Using Established Methods: Cite the creators of methodologies you employ in your survey.
6. Supporting Arguments: Cite sources that back up your conclusions and arguments.
7. Suggesting Future Research: Reference studies related to proposed future research directions.

Ensure sufficient cites will be collected for all of these categories, and no categories are missed.
You will be given access to the Semantic Scholar API; only add citations that you have found using the API.
Aim to discuss a broad range of relevant papers, not just the most popular ones.
Make sure not to copy verbatim from prior literature to avoid plagiarism.
You will have {total_rounds} rounds to add to the references but do not need to use them all.

DO NOT ADD A CITATION THAT ALREADY EXISTS!"""

    citation_first_prompt_template = """Round {current_round}/{total_rounds}:

You planned and executed the following idea:
```markdown
{Idea}
```

You produced the following report:
```markdown
{report}
```

Your current list of citations is:
```
{citations}
```

Identify the most important citation that you still need to add, and the query to find the paper.

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason and identify which citations are missing.
If no more citations are needed, add "No more citations needed" to your thoughts.
Do not add "No more citations needed" if you are adding citations this round.

In <JSON>, respond in JSON format with the following fields:
- "Description": The purpose of the desired citation and a brief description of what you are looking for.
- "Query": The search query to find the paper (e.g., attention is all you need).
This JSON will be automatically parsed, so ensure the format is precise."""

    citation_second_prompt_template = """Search has recovered the following articles:

{papers}

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the search results and identify which citation(s) best fit your paper.
If none are appropriate or would contribute significantly to the write-up, add "Do not add any" to your thoughts.
Do not select papers that are already in the `references.bib` file, or if the same citation exists under a different name.

In <JSON>, respond in JSON format with the following fields:
- "Selected": A list of integer indices for the selected papers, for example [0, 1]. Do not use quotes for the indices, e.g. "['0', '1']" is invalid.
- "Description": Update the previous description of the citation(s) with the additional context. This should be a brief description of the work(s), their relevance, and where in a paper these should be cited.
This JSON will be automatically parsed, so ensure the format is precise."""

    try:
        text, msg_history = get_response_from_llm(
            msg=citation_first_prompt_template.format(
                current_round=current_round + 1,
                total_rounds=total_rounds,
                Idea=idea_text,
                report=report,
                citations=citations,
            ),
            client=client,
            model=model,
            system_message=citation_system_msg_template.format(
                total_rounds=total_rounds
            ),
            msg_history=msg_history,
            print_debug=False,
        )
        if "No more citations needed" in text:
            print("No more citations needed.")
            return None, True

        json_output = extract_json_between_markers(text)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        query = json_output["Query"]
        papers = search_for_papers(query)
    except Exception:
        print("EXCEPTION in get_citation_addition (initial search):")
        print(traceback.format_exc())
        return None, False

    if papers is None:
        print("No papers found.")
        return None, False

    paper_strings = []
    for i, paper in enumerate(papers):
        paper_strings.append(
            "{i}: {title}. {authors}. {venue}, {year}.\nAbstract: {abstract}".format(
                i=i,
                title=paper["title"],
                authors=paper["authors"],
                venue=paper["venue"],
                year=paper["year"],
                abstract=paper["abstract"],
            )
        )
    papers_str = "\n\n".join(paper_strings)

    try:
        text, msg_history = get_response_from_llm(
            msg=citation_second_prompt_template.format(
                papers=papers_str,
                current_round=current_round + 1,
                total_rounds=total_rounds,
            ),
            client=client,
            model=model,
            system_message=citation_system_msg_template.format(
                total_rounds=total_rounds
            ),
            msg_history=msg_history,
            print_debug=False,
        )
        if "Do not add any" in text:
            print("Do not add any.")
            return None, False

        json_output = extract_json_between_markers(text)
        assert json_output is not None, "Failed to extract JSON from LLM output"
        desc = json_output["Description"]
        selected_papers = str(json_output["Selected"])

        if selected_papers != "[]":
            selected_indices = []
            for x in selected_papers.strip("[]").split(","):
                x_str = x.strip().strip('"').strip("'")
                if x_str:
                    selected_indices.append(int(x_str))
            assert all(
                [0 <= i < len(papers) for i in selected_indices]
            ), "Invalid paper index"
            bibtexs = [papers[i]["citationStyles"]["bibtex"] for i in selected_indices]

            cleaned_bibtexs = []
            for bibtex in bibtexs:
                newline_index = bibtex.find("\n")
                cite_key_line = bibtex[:newline_index]
                cite_key_line = remove_accents_and_clean(cite_key_line)
                cleaned_bibtexs.append(cite_key_line + bibtex[newline_index:])
            bibtexs = cleaned_bibtexs

            bibtex_string = "\n".join(bibtexs)
        else:
            return None, False

    except Exception:
        print("EXCEPTION in get_citation_addition (selecting papers):")
        print(traceback.format_exc())
        return None, False

    references_format = """% {description}
{bibtex}"""

    references_prompt = references_format.format(bibtex=bibtex_string, description=desc)
    return references_prompt, False


# Using a template string to allow injection of the {page_limit} argument
writeup_system_message_template = """You are an ambitious AI researcher who is looking to publish a paper that will contribute significantly to the field.
Ensure that the paper is scientifically accurate, objective, and truthful. Accurately report the experimental results, even if they are negative or inconclusive.
You are planning to submit to a top-tier ML conference, which has guidelines:
- The main paper is limited to {page_limit} pages, including all figures and tables, but excluding references, the impact statement, and optional appendices. In general, try to use the available space and include all relevant information.
- The main paper should be double-column format, while the appendices can be in single-column format. When in double column format, make sure that tables and figures are correctly placed.
- Do not change the overall style which is mandated by the conference. Keep to the current method of including the references.bib file.
- Do not remove the \\graphicspath directive or no figures will be found.

Here are some tips for each section of the paper:

- **Title**:
  - Title should be catchy and informative. It should give a good idea of what the paper is about.
  - Try to keep it under 2 lines.

- **Abstract**:
  - TL;DR of the paper.
  - What are we trying to do and why is it relevant?
  - Make sure the abstract reads smoothly and is well-motivated. This should be one continuous paragraph.

- **Introduction**:
  - Longer version of the Abstract, i.e., an overview of the entire paper.
  - Provide context to the study and explain its relevance.
  - If results are inconclusive or negative, present them frankly; if they are positive, you may highlight how the approach effectively addresses the research question or problem.
  - Summarize your contributions, highlighting pertinent findings, insights, or proposed methods.

- **Related Work**:
  - Academic siblings of our work, i.e., alternative attempts in literature at trying to address the same or similar problems.
  - Compare and contrast their approach with yours, noting key differences or similarities.
  - Ensure proper citations are provided.

- **Background**:
  - Present foundational concepts or prior work needed to understand your method.
  - This should include necessary definitions, the problem setting, or relevant theoretical constructs.

- **Method**:
  - Clearly detail what you propose to do and why. If your study aims to address certain hypotheses, describe them and how your method is constructed to test them.
  - If results are negative or inconclusive, you may suggest improvements or discuss possible causes.

- **Experimental Setup**:
  - Explain how you tested your method or hypothesis.
  - Describe necessary details such as data, environment, and baselines, but omit hardware details unless explicitly mentioned.

- **Experiments**:
  - Present the results truthfully according to the data you have. If outcomes are not as expected, discuss it transparently.
  - Include comparisons to baselines if available, and only include analyses supported by genuine data.
  - Try to include all relevant plots and tables. Consider combining multiple plots into one figure if they are related.

- **Conclusion**:
  - Summarize the entire paper, including key strengths or findings.
  - If results are strong, highlight how they might address the research problem.
  - If results are negative or inconclusive, highlight potential improvements or reasons and propose future directions.

- **Appendix**:
  - Place for supplementary material that did not fit in the main paper.

Ensure you are always writing good compilable LaTeX code. Common mistakes that should be fixed include:
- LaTeX syntax errors (unenclosed math, unmatched braces, etc.).
- Duplicate figure labels or references.
- Unescaped special characters: & % $ # _ {{ }} ~ ^ \\
- Proper table/figure closure.
- Do not hallucinate new citations or any results not in the logs.

When returning final code, place it in fenced triple backticks with 'latex' syntax highlighting.
"""

writeup_prompt = """Your goal is to write up the following idea:

```markdown
{idea_text}
```

We have the following experiment summaries (JSON):
```json
{summaries}
```

We also have a script used to produce the final plots (use this to see how the plots are generated and what names are used in the legend):
```python
{aggregator_code}
```
Please also consider which plots should naturally be grouped together as subfigures.

Available plots for the writeup (use these filenames):
```
{plot_list}
```

We also have VLM-based figure descriptions:
```
{plot_descriptions}
```

We also have structured manuscript state and evidence bindings:
```
{structured_context}
```

Your current progress on the LaTeX write-up is:
```latex
{latex_writeup}
```

Produce the final version of the LaTeX manuscript now, ensuring the paper is coherent, concise, and reports results accurately.
Return the entire file in full, with no unfilled placeholders!
This must be an acceptable complete LaTeX writeup.

Please provide the updated LaTeX code for 'template.tex', wrapped in triple backticks
with "latex" syntax highlighting, like so:

```latex
<UPDATED LATEX CODE>
```
"""


def perform_writeup(
    base_folder,
    citations_text=None,
    no_writing=False,
    num_cite_rounds=20,
    small_model=os.environ.get("ZHIPU_DEFAULT_MODEL", "glm-4-flash"),
    big_model="o1-2024-12-17",
    n_writeup_reflections=3,
    page_limit=8,
    writing_profile: str | None = None,
    writing_audit_rounds: int = 0,
    target_venue: str | None = None,
    strict_guardrails: bool = False,
    guardrail_repair_rounds: int = 1,
):
    compile_attempt = 0
    base_pdf_file = osp.join(base_folder, f"{osp.basename(base_folder)}")
    latex_folder = osp.join(base_folder, "latex")
    resolved_profile = _resolve_writing_profile(writing_profile)
    profile_system_guidance = render_writing_profile_system_guidance(resolved_profile)
    profile_self_checks = render_writing_profile_self_checks(resolved_profile)
    venue_checklist = render_venue_checklist(target_venue)
    citation_integrity_rules = render_citation_integrity_rules()
    humanizer_style_rules = render_humanizer_style_rules(resolved_profile)
    skill_pack_guidance = render_writing_skill_pack(target_venue=target_venue)
    writing_audit_rounds = max(0, int(writing_audit_rounds))
    guardrail_repair_rounds = max(0, int(guardrail_repair_rounds))
    audits_dir = osp.join(base_folder, "writing_audits")
    os.makedirs(audits_dir, exist_ok=True)
    print(f"Using writing profile: {resolved_profile}")
    if target_venue:
        print(f"Target venue guardrails: {target_venue}")
    if writing_audit_rounds > 0:
        print(f"Writing audit rounds enabled: {writing_audit_rounds}")
    if strict_guardrails:
        print("Strict writeup guardrails enabled.")
        print(f"Guardrail repair rounds: {guardrail_repair_rounds}")

    # Cleanup any previous latex folder and pdf
    if osp.exists(latex_folder):
        shutil.rmtree(latex_folder)
    # if osp.exists(pdf_file):
    #     os.remove(pdf_file)

    try:
        # Load idea text
        idea_text = ""
        research_idea_path = osp.join(base_folder, "research_idea.md")
        if osp.exists(research_idea_path):
            with open(research_idea_path, "r") as f_idea:
                idea_text = f_idea.read()
        else:
            idea_md_path = osp.join(base_folder, "idea.md")
            if osp.exists(idea_md_path):
                with open(idea_md_path, "r") as f_idea:
                    idea_text = f_idea.read()

        # Load summaries
        summary_files = [
            ("baseline_summary.json", "BASELINE_SUMMARY"),
            ("research_summary.json", "RESEARCH_SUMMARY"),
            ("ablation_summary.json", "ABLATION_SUMMARY"),
        ]
        loaded_summaries = {}
        run_dirs = list(
            iter_bfts_run_dirs(base_folder, logs_subdir="logs", descending=True)
        )
        for filename, key in summary_files:
            summary = {}
            for run_dir in run_dirs:
                path = run_dir / filename
                if not path.exists():
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    break
                except json.JSONDecodeError:
                    print(
                        f"Warning: {path} is not valid JSON. Using empty data for {key}."
                    )
                    summary = {}
                    break
                except OSError:
                    summary = {}
                    break
            loaded_summaries[key] = summary

        # Convert them to one big JSON string for context
        combined_summaries_str = json.dumps(loaded_summaries, indent=2)

        # Prepare a new fresh latex folder
        if not osp.exists(osp.join(latex_folder, "template.tex")):
            shutil.copytree(
                "ai_scientist/blank_icml_latex", latex_folder, dirs_exist_ok=True
            )

        writeup_file = osp.join(latex_folder, "template.tex")
        with open(writeup_file, "r") as f:
            writeup_text = f.read()

        # Gather plot filenames from figures/ folder
        figures_dir = osp.join(base_folder, "figures")
        plot_names = []
        if osp.exists(figures_dir):
            for fplot in os.listdir(figures_dir):
                if fplot.lower().endswith(".png"):
                    plot_names.append(fplot)

        # Load aggregator script to include in the prompt
        aggregator_path = osp.join(base_folder, "auto_plot_aggregator.py")
        aggregator_code = ""
        if osp.exists(aggregator_path):
            with open(aggregator_path, "r") as fa:
                aggregator_code = fa.read()
        else:
            aggregator_code = "No aggregator script found."

        if no_writing:
            compile_latex(latex_folder, base_pdf_file + ".pdf")
            return osp.exists(base_pdf_file + ".pdf")

        if citations_text:
            with open(writeup_file, "r", encoding="utf-8") as f:
                seeded_content = f.read()
            pattern_end = r"\end{filecontents}"
            if pattern_end in seeded_content and citations_text not in seeded_content:
                seeded_content = seeded_content.replace(
                    pattern_end, f"\n{citations_text}{pattern_end}"
                )
                with open(writeup_file, "w", encoding="utf-8") as f:
                    f.write(seeded_content)

        # Run small model for citation additions
        client, client_model = create_client(small_model)
        for round_idx in range(num_cite_rounds):
            with open(writeup_file, "r") as f:
                writeup_text = f.read()
            try:
                references_bib = re.search(
                    r"\\begin{filecontents}{references.bib}(.*?)\\end{filecontents}",
                    writeup_text,
                    re.DOTALL,
                )
                if references_bib is None:
                    raise ValueError("No references.bib found in template.tex")
                citations_text = references_bib.group(1)
                context_for_citation = (combined_summaries_str, citations_text)

                addition, done = get_citation_addition(
                    client,
                    client_model,
                    context_for_citation,
                    round_idx,
                    num_cite_rounds,
                    idea_text,
                )
                if done:
                    break

                if addition is not None:
                    # Simple check to avoid duplicating the same title
                    title_match = re.search(r" title = {(.*?)}", addition)
                    if title_match:
                        new_title = title_match.group(1).lower()
                        existing_titles = re.findall(
                            r" title = {(.*?)}", citations_text
                        )
                        existing_titles = [t.lower() for t in existing_titles]
                        if new_title not in existing_titles:
                            pattern_end = r"\end{filecontents}"
                            revised = writeup_text.replace(
                                pattern_end, f"\n{addition}{pattern_end}"
                            )
                            with open(writeup_file, "w") as fo:
                                fo.write(revised)
            except Exception:
                print("EXCEPTION in perform_writeup (citation round):")
                print(traceback.format_exc())
                continue

        # Generate VLM-based descriptions but do not overwrite plot_names
        try:
            vlm_client, vlm_model = create_vlm_client(small_model)
            desc_map = {}
            for pf in plot_names:
                ppath = osp.join(figures_dir, pf)
                if not osp.exists(ppath):
                    continue
                img_dict = {
                    "images": [ppath],
                    "caption": "No direct caption",
                }
                review_data = generate_vlm_img_review(img_dict, vlm_model, vlm_client)
                if review_data:
                    desc_map[pf] = review_data.get(
                        "Img_description", "No description found"
                    )
                else:
                    desc_map[pf] = "No description found"

            # Prepare a string listing all figure descriptions in order
            plot_descriptions_list = []
            for fname in plot_names:
                desc_text = desc_map.get(fname, "No description found")
                plot_descriptions_list.append(f"{fname}: {desc_text}")
            plot_descriptions_str = "\n".join(plot_descriptions_list)
        except Exception:
            print("EXCEPTION in VLM figure description generation:")
            print(traceback.format_exc())
            plot_descriptions_str = "No descriptions available."

        # Construct final prompt for big model, placing the figure descriptions alongside the plot list
        manuscript_state = load_contract_artifact(
            base_folder,
            "manuscript_state",
            default={},
        )
        structured_context = render_manuscript_prompt_context(
            manuscript_state if isinstance(manuscript_state, dict) else {}
        )
        big_model_system_message = (
            writeup_system_message_template.format(page_limit=page_limit)
            + "\n\n"
            + profile_system_guidance
            + "\n\n"
            + skill_pack_guidance
            + "\n\n"
            + venue_checklist
            + "\n\n"
            + citation_integrity_rules
            + "\n\n"
            + humanizer_style_rules
        )
        big_client, big_client_model = create_client(big_model)
        with open(writeup_file, "r") as f:
            writeup_text = f.read()

        combined_prompt = writeup_prompt.format(
            idea_text=idea_text,
            summaries=combined_summaries_str,
            aggregator_code=aggregator_code,
            plot_list=", ".join(plot_names),
            latex_writeup=writeup_text,
            plot_descriptions=plot_descriptions_str,
            structured_context=structured_context,
        )

        response, msg_history = get_response_from_llm(
            msg=combined_prompt,
            client=big_client,
            model=big_client_model,
            system_message=big_model_system_message,
            print_debug=False,
        )

        latex_code_match = re.search(r"```latex(.*?)```", response, re.DOTALL)
        if not latex_code_match:
            return False
        updated_latex_code = latex_code_match.group(1).strip()
        with open(writeup_file, "w") as f:
            f.write(updated_latex_code)

        # Multiple reflection loops on the final LaTeX
        for i in range(n_writeup_reflections):
            with open(writeup_file, "r") as f:
                current_latex = f.read()

            audit_prompt_block = "Audit skipped for this round."
            if i < writing_audit_rounds:
                try:
                    audit_result = run_writing_audit(
                        idea_text=idea_text,
                        summaries_json=combined_summaries_str,
                        current_latex=current_latex,
                        client=big_client,
                        model=big_client_model,
                        system_message=big_model_system_message,
                        profile_guidance=profile_system_guidance,
                        profile_self_checks=profile_self_checks,
                        venue_checklist=venue_checklist,
                        citation_integrity_rules=citation_integrity_rules,
                        humanizer_style_checks=humanizer_style_rules,
                    )
                    audit_prompt_block = format_writing_audit_for_prompt(audit_result)
                    with open(
                        osp.join(audits_dir, f"audit_round_{i + 1}.json"),
                        "w",
                        encoding="utf-8",
                    ) as f_audit:
                        json.dump(audit_result, f_audit, indent=2, ensure_ascii=False)
                except Exception:
                    print(f"EXCEPTION in writing audit round {i + 1}:")
                    print(traceback.format_exc())
                    audit_prompt_block = "Audit failed due to runtime exception."

            # Check for unused or invalid figure references
            referenced_figs_temp = re.findall(
                r"\\includegraphics(?:\[[^\]]*\])?{([^}]+)}", current_latex
            )
            used_figs = set(os.path.basename(fig) for fig in referenced_figs_temp)
            all_figs = set(plot_names)
            unused_figs = all_figs - used_figs
            invalid_figs = used_figs - all_figs

            # Compile current version before reflection
            compile_latex(latex_folder, base_pdf_file + f"_{compile_attempt}.pdf")
            compile_attempt += 1
            print(f"Compiled {base_pdf_file}_{compile_attempt}.pdf")

            # Detect where "Impact Statement" appears
            impact_loc = detect_pages_before_impact(latex_folder)
            if impact_loc is not None:
                page_num, line_num = impact_loc
                reflection_page_info = (
                    f"\nCurrently, 'Impact Statement' begins on page {page_num}, approximately on line {line_num}. "
                    f"The page limit is {page_limit}, which is before the Impact Statement. "
                    f"Papers often look more professional if the main text is near or just under {page_limit} pages in length.\n"
                )
            else:
                reflection_page_info = "\nCould not detect 'Impact Statement' page (compilation or detection failed).\n"

            check_output = run_chktex(writeup_file)
            citation_consistency_report = build_citation_consistency_report(current_latex)
            submission_guardrail_report = build_submission_guardrail_report(
                current_latex, target_venue
            )

            reflection_prompt = f"""
Now let's reflect and identify any issues (including but not limited to):
1) Are there any LaTeX syntax errors or style violations we can fix? Refer to the chktex output below.
2) Is the writing clear, and scientifically rigorous?
3) Have we included all relevant details from the summaries without hallucinating?
4) The following figures are available in the folder but not used in the LaTeX: {sorted(unused_figs)}
5) The following figure references in the LaTeX do not match any actual file: {sorted(invalid_figs)}
{reflection_page_info}
chktex results:
```
{check_output}
```
{profile_self_checks}
{venue_checklist}
{citation_integrity_rules}
{humanizer_style_rules}
{citation_consistency_report}
{submission_guardrail_report}
Structured writing audit findings:
```
{audit_prompt_block}
```

Please provide a revised complete LaTeX in triple backticks, or repeat the same if no changes are needed.
Return the entire file in full, with no unfilled placeholders!
This must be an acceptable complete LaTeX writeup.
Do not hallucinate any details!

If you believe you are done, simply say: "I am done".
"""

            reflection_response, msg_history = get_response_from_llm(
                msg=reflection_prompt,
                client=big_client,
                model=big_client_model,
                system_message=big_model_system_message,
                msg_history=msg_history,
                print_debug=False,
            )

            if "I am done" in reflection_response:
                print(
                    "LLM indicated it is done with reflections. Exiting reflection loop."
                )
                break

            reflection_code_match = re.search(
                r"```latex(.*?)```", reflection_response, re.DOTALL
            )
            if reflection_code_match:
                reflected_latex_code = reflection_code_match.group(1).strip()
                if reflected_latex_code != current_latex:
                    final_text = reflected_latex_code
                    cleanup_map = {
                        "</end": r"\\end",
                        "</begin": r"\\begin",
                        "’": "'",
                    }
                    for bad_str, repl_str in cleanup_map.items():
                        final_text = final_text.replace(bad_str, repl_str)
                    final_text = re.sub(r"(\d+(?:\.\d+)?)%", r"\1\\%", final_text)

                    with open(writeup_file, "w") as fo:
                        fo.write(final_text)

                    compile_latex(
                        latex_folder, base_pdf_file + f"_{compile_attempt}.pdf"
                    )
                    compile_attempt += 1
                    print(f"Compiled {base_pdf_file}_{compile_attempt}.pdf")
                else:
                    print(f"No changes in reflection step {i+1}.")
                    break
            else:
                print(f"No valid LaTeX code block found in reflection step {i+1}.")
                break

        final_pdf_exists = osp.exists(base_pdf_file + f"_{compile_attempt-1}.pdf")
        if not final_pdf_exists:
            final_pdf_exists = osp.exists(base_pdf_file + ".pdf")
        with open(writeup_file, "r", encoding="utf-8", errors="ignore") as f_final:
            final_latex = f_final.read()
        final_guardrail_findings = collect_guardrail_findings(
            final_latex, target_venue
        )
        final_guardrail_report = build_submission_guardrail_report(
            final_latex, target_venue
        )

        if strict_guardrails and has_blocking_guardrail_violations(
            final_guardrail_findings,
            allow_placeholder_citations=False,
            require_venue_sections=True,
        ):
            blocking_reasons = list_blocking_guardrail_reasons(
                final_guardrail_findings,
                allow_placeholder_citations=False,
                require_venue_sections=True,
            )
            repair_plan = build_guardrail_repair_plan(
                final_guardrail_findings,
                target_venue,
            )
            for repair_idx in range(guardrail_repair_rounds):
                repair_prompt = f"""
The manuscript failed strict submission guardrails.
Fix all blocking issues listed below while preserving true experimental content and citations:
```
{final_guardrail_report}
```
Blocking reasons:
{", ".join(blocking_reasons) if blocking_reasons else "unknown"}

Actionable repair plan:
{repair_plan}

Hard requirements:
- Do not invent citations, results, or claims.
- Keep references.bib consistent with in-text citations.
- Add or improve required venue-aligned sections when missing.
- Return the entire updated LaTeX file in one complete ```latex``` block.
"""
                repair_response, _ = get_response_from_llm(
                    msg=repair_prompt,
                    client=big_client,
                    model=big_client_model,
                    system_message=big_model_system_message,
                    print_debug=False,
                )
                repair_match = re.search(
                    r"```latex(.*?)```", repair_response, re.DOTALL
                )
                if not repair_match:
                    break
                repaired_latex = repair_match.group(1).strip()
                if repaired_latex == final_latex:
                    break
                with open(writeup_file, "w", encoding="utf-8") as f_out:
                    f_out.write(repaired_latex)
                compile_latex(
                    latex_folder,
                    base_pdf_file + f"_guardrail_repair_{repair_idx + 1}.pdf",
                )
                final_latex = repaired_latex
                final_guardrail_findings = collect_guardrail_findings(
                    final_latex, target_venue
                )
                final_guardrail_report = build_submission_guardrail_report(
                    final_latex, target_venue
                )
                round_reasons = list_blocking_guardrail_reasons(
                    final_guardrail_findings,
                    allow_placeholder_citations=False,
                    require_venue_sections=True,
                )
                round_repair_plan = build_guardrail_repair_plan(
                    final_guardrail_findings,
                    target_venue,
                )
                with open(
                    osp.join(audits_dir, f"guardrail_repair_round_{repair_idx + 1}.json"),
                    "w",
                    encoding="utf-8",
                ) as f_repair:
                    json.dump(
                        {
                            "findings": final_guardrail_findings,
                            "reasons": round_reasons,
                            "repair_plan": round_repair_plan,
                        },
                        f_repair,
                        indent=2,
                        ensure_ascii=False,
                    )
                if not has_blocking_guardrail_violations(
                    final_guardrail_findings,
                    allow_placeholder_citations=False,
                    require_venue_sections=True,
                ):
                    break

        with open(
            osp.join(audits_dir, "final_guardrail_report.json"),
            "w",
            encoding="utf-8",
        ) as f_report_json:
            json.dump(final_guardrail_findings, f_report_json, indent=2, ensure_ascii=False)
        with open(
            osp.join(audits_dir, "final_guardrail_report.txt"),
            "w",
            encoding="utf-8",
        ) as f_report_txt:
            f_report_txt.write(final_guardrail_report + "\n")

        if strict_guardrails and has_blocking_guardrail_violations(
            final_guardrail_findings,
            allow_placeholder_citations=False,
            require_venue_sections=True,
        ):
            reasons = list_blocking_guardrail_reasons(
                final_guardrail_findings,
                allow_placeholder_citations=False,
                require_venue_sections=True,
            )
            with open(
                osp.join(audits_dir, "final_guardrail_failure_reasons.json"),
                "w",
                encoding="utf-8",
            ) as f_reasons:
                json.dump(
                    {
                        "reasons": reasons,
                        "findings": final_guardrail_findings,
                        "repair_plan": build_guardrail_repair_plan(
                            final_guardrail_findings, target_venue
                        ),
                    },
                    f_reasons,
                    indent=2,
                    ensure_ascii=False,
                )
            print("Strict guardrail check failed. Final findings:")
            print(final_guardrail_report)
            print("Blocking reasons:", ", ".join(reasons) if reasons else "unknown")
            return False

        return final_pdf_exists

    except Exception:
        print("EXCEPTION in perform_writeup:")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    require_login("论文写作(perform_writeup)")

    parser = argparse.ArgumentParser(description="Perform writeup for a project")
    parser.add_argument("--folder", type=str, help="Project folder", required=True)
    parser.add_argument("--no-writing", action="store_true", help="Only generate")
    parser.add_argument("--num-cite-rounds", type=int, default=20)
    parser.add_argument(
        "--model",
        type=str,
        default="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        choices=AVAILABLE_LLMS,
        help="Model to use for citation collection (small model).",
    )
    parser.add_argument(
        "--big-model",
        type=str,
        default="o1-2024-12-17",
        choices=AVAILABLE_LLMS,
        help="Model to use for final writeup (big model).",
    )
    parser.add_argument(
        "--writeup-reflections",
        type=int,
        default=3,
        help="Number of reflection steps for the final LaTeX writeup.",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=8,
        help="Target page limit for the main paper (excluding references, impact statement, etc.)",
    )
    parser.add_argument(
        "--writing-profile",
        type=str,
        default=os.environ.get(
            "AI_SCIENTIST_WRITING_PROFILE", DEFAULT_WRITING_PROFILE
        ),
        choices=list_writing_profiles(),
        help="Prompt writing profile used to guide style and self-checks.",
    )
    parser.add_argument(
        "--writing-audit-rounds",
        type=int,
        default=0,
        help="Number of structured writing audit rounds injected into reflection loop.",
    )
    parser.add_argument(
        "--target-venue",
        type=str,
        default=None,
        help="Optional target venue used for submission checklist guardrails.",
    )
    parser.add_argument(
        "--strict-guardrails",
        action="store_true",
        help="Fail writeup if final citation/section guardrails are not satisfied.",
    )
    parser.add_argument(
        "--guardrail-repair-rounds",
        type=int,
        default=1,
        help="Automatic repair rounds attempted before strict guardrail failure.",
    )
    args = parser.parse_args()

    try:
        success = perform_writeup(
            base_folder=args.folder,
            no_writing=args.no_writing,
            num_cite_rounds=args.num_cite_rounds,
            small_model=args.model,
            big_model=args.big_model,
            n_writeup_reflections=args.writeup_reflections,
            page_limit=args.page_limit,
            writing_profile=args.writing_profile,
            writing_audit_rounds=args.writing_audit_rounds,
            target_venue=args.target_venue,
            strict_guardrails=args.strict_guardrails,
            guardrail_repair_rounds=args.guardrail_repair_rounds,
        )
        if not success:
            print("Writeup process did not complete successfully.")
    except Exception:
        print("EXCEPTION in main:")
        print(traceback.format_exc())
