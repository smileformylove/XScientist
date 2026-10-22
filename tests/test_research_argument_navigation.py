from __future__ import annotations

from html.parser import HTMLParser
import json
import subprocess
import unittest

from ai_scientist.protocol.research_vcs import build_research_object
from tests.node_test_support import NODE_EXECUTABLE
from tests import test_research_argument as argument_tests
from xscientist.research_argument import build_research_argument_view
from xscientist.research_dag import render_research_dag_html


class _Scripts(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: dict[str, str] = {}
        self.current: str | None = None
        self.elements: dict[str, dict] = {}
        self.graph_controls: list[dict] = []
        self.disclosure_ids: list[str] = []
        self.legend_ancestor_ids: list[str] = []
        self.stack: list[tuple[str, str | None]] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        identifier = attrs.get("id")
        if identifier:
            self.elements[identifier] = {
                "tag": tag,
                "attrs": attrs,
                "ancestor_ids": [item[1] for item in self.stack if item[1]],
            }
        if "data-graph-control" in attrs:
            self.graph_controls.append(attrs)
        if "data-compact-disclosure" in attrs:
            self.disclosure_ids.append(identifier)
        if "rendering-legend" in attrs.get("class", "").split():
            self.legend_ancestor_ids = [item[1] for item in self.stack if item[1]]
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append((tag, identifier))
        if tag == "script":
            self.current = attrs.get("id", "executable")
            self.scripts[self.current] = ""

    def handle_data(self, data):
        if self.current is not None:
            self.scripts[self.current] += data

    def handle_endtag(self, tag):
        if tag == "script":
            self.current = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break


# This DOM stub exercises generated JavaScript, not browser layout or rendering.
_NODE_HARNESS = r"""
const vm = require('node:vm'), assert = require('node:assert/strict');
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
class Element {
  constructor(id) { this.id=id; this.value=''; this.hidden=false; this.textContent=''; this.html=''; this.events={}; this.children=[]; this.writes=0; this.dataset={}; this.className=''; this.classList={toggle(){}}; }
  set innerHTML(value) {
    this.html=value; this.writes++; this.children=[];
    const tags=/<(button|g)\b([^>]*)>/g;
    for (const match of value.matchAll(tags)) {
      const attrs=Object.fromEntries([...match[2].matchAll(/([\w-]+)="([^"]*)"/g)].map(row=>[row[1],row[2]]));
      const classes=(attrs.class||'').split(/\s+/);
      if (!classes.includes('node-jump') && !classes.includes('node')) continue;
      const child=new Element(''); child.classes=classes;
      child.dataset={nodeId:attrs['data-node-id'],id:attrs['data-id']}; this.children.push(child);
    }
  }
  get innerHTML() { return this.html; }
  querySelectorAll(selector) {
    if(this.id==='argument-view') return elements['registered-claims'].querySelectorAll(selector).concat(elements['advisory-routes'].querySelectorAll(selector));
    return this.children.filter(child=>(child.classes||[]).includes(selector.slice(1)));
  }
  addEventListener(type, callback) { (this.events[type]||=[]).push(callback); }
  fire(type) { for(const callback of this.events[type]||[]) callback({target:this,preventDefault(){}}); }
  insertAdjacentHTML() {}
}
const elements=Object.fromEntries(Object.keys(input.elements).map(id=>[id,new Element(id)]));
elements.view.value='overview'; elements.source.value='all';
for(const [id,data] of Object.entries(input.scripts)) if(id!=='executable') {elements[id]=new Element(id);elements[id].textContent=data;}
for(const id of input.disclosure_ids) {
  elements[id]=new Element(id);
  elements[id].open=Object.hasOwn(input.elements[id].attrs,'open');
}
const graphControls=input.graph_controls.map((attrs,index)=>attrs.id?elements[attrs.id]:new Element('filter-'+index));
const disclosures=input.disclosure_ids.map(id=>elements[id]);
const document={getElementById:id=>elements[id],querySelectorAll:selector=>selector==='[data-graph-control]'?graphControls:selector==='[data-compact-disclosure]'?disclosures:elements.canvas.querySelectorAll(selector)};
const sandbox={document,assert,elements,graphControls,targets:input.targets,console};
if(Object.hasOwn(input,'compact')) {
  const mediaStub={media:'(max-width:600px)',matches:input.compact,listeners:[],
    addEventListener(type,callback) { assert.equal(type,'change'); this.listeners.push(callback); },
    setMatches(matches) {
      if(this.matches===matches)return;
      this.matches=matches;
      for(const callback of this.listeners)callback({matches,media:this.media});
    }};
  sandbox.mediaStub=mediaStub;
  sandbox.matchMedia=query=>{assert.equal(query,mediaStub.media);return mediaStub;};
}
vm.createContext(sandbox);
vm.runInContext(input.scripts.executable,sandbox,{timeout:5000});
vm.runInContext(input.checks,sandbox,{timeout:5000});
console.log('PASS generated-page navigation fixture');
"""


def _navigation_fixture() -> tuple[_Scripts, dict[str, str]]:
    fixture = argument_tests.ResearchArgumentProjectionTests
    graph, objects, closure = fixture._claim_fixture(include_role_binding=False)
    review = build_research_object(
        kind="review", payload={"summary": "Synthetic independent review"},
        state="completed", actor={"actor_id": "navigation-review", "authority": "independent_evaluator"},
    )
    gate = build_research_object(
        kind="gate_decision", payload={"decision": "hold", "claim_promotion_allowed": False},
        state="completed", relations=[{"type": "evaluates", "target": review["object_id"]}],
    )
    for item in (review, gate):
        objects.append(item)
        fixture._append_research_node(graph, item, summary="Synthetic " + item["kind"])
    graph["edges"].append({"source": review["object_id"], "target": gate["object_id"],
                           "type": "evaluates", "role": "", "category": "verification"})
    closure["claims"][0]["gate_ids"] = [gate["object_id"]]
    graph["claim_insights"] = []
    graph["strategy_summary"] = {"open_anomaly_count": 0}
    graph["integrity"] = {"is_dag": True, "topological_order": [node["id"] for node in graph["nodes"]]}
    for node in graph["nodes"]:
        node.setdefault("summary", "Synthetic navigation node")
        node.setdefault("phase", 6)
        node.setdefault("layer", "evidence")
        node.setdefault("proof", {"level": "recorded", "checks": [], "blockers": [], "warnings": [], "contested": False})
    fixture._rebind_closure(graph, closure)
    argument = build_research_argument_view(graph, object_rows=objects, closure=closure)
    targets = {"support": objects[1]["object_id"], "review": review["object_id"],
               "gate": gate["object_id"], "claim": objects[0]["object_id"]}
    card = argument["registered_claims"][0]
    for key, field in (("support", "direct_supporting_ids"), ("review", "independent_review_ids"), ("gate", "gate_ids")):
        if targets[key] not in card[field]:
            raise AssertionError("fixture did not project " + key)
    parser = _Scripts()
    parser.feed(render_research_dag_html(graph, argument_view=argument))
    return parser, targets


class ResearchDagDisclosureStructureTests(unittest.TestCase):
    def test_native_disclosures_wrap_filters_and_legend_but_not_view(self) -> None:
        page, _ = _navigation_fixture()
        self.assertEqual(page.disclosure_ids, ["filter-disclosure", "legend-disclosure"])
        for identifier in page.disclosure_ids:
            self.assertEqual(page.elements[identifier]["tag"], "details")
            self.assertIn("open", page.elements[identifier]["attrs"])
        self.assertIn("data-graph-control", page.elements["filter-disclosure"]["attrs"])
        self.assertNotIn("data-graph-control", page.elements["legend-disclosure"]["attrs"])
        self.assertNotIn("filter-disclosure", page.elements["view"]["ancestor_ids"])
        self.assertNotIn("legend-disclosure", page.elements["view"]["ancestor_ids"])
        self.assertIn("legend-disclosure", page.legend_ancestor_ids)
        for identifier in ("source", "search", "layer", "kind", "proof"):
            self.assertIn("filter-disclosure", page.elements[identifier]["ancestor_ids"])


@unittest.skipUnless(NODE_EXECUTABLE, "Node.js is required for navigation tests")
class ResearchArgumentNavigationTests(unittest.TestCase):
    def _check(self, checks: str, *, compact: bool | None = None) -> None:
        page, targets = _navigation_fixture()
        payload = {"scripts": page.scripts, "targets": targets, "checks": checks,
                   "elements": page.elements, "graph_controls": page.graph_controls,
                   "disclosure_ids": page.disclosure_ids}
        if compact is not None:
            payload["compact"] = compact
        result = subprocess.run(
            [NODE_EXECUTABLE, "-e", _NODE_HARNESS],
            input=json.dumps(payload),
            text=True, capture_output=True, check=False, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "PASS generated-page navigation fixture")

    def test_disclosures_keep_native_defaults_without_match_media(self) -> None:
        self._check(r"""
assert.equal(typeof matchMedia,'undefined');
for(const id of ['filter-disclosure','legend-disclosure'])assert.equal(elements[id].open,true);
render();
for(const id of ['filter-disclosure','legend-disclosure'])assert.equal(elements[id].open,true);
""")

    def test_disclosures_start_open_on_desktop_and_closed_on_mobile(self) -> None:
        for compact in (False, True):
            with self.subTest(compact=compact):
                self._check(r"""
for(const id of ['filter-disclosure','legend-disclosure'])assert.equal(elements[id].open,!mediaStub.matches);
assert.equal(mediaStub.listeners.length,1);
assert.equal(elements['filter-disclosure'].hidden,false);
assert.equal(view.hidden,false);
""", compact=compact)

    def test_disclosures_sync_when_crossing_the_compact_breakpoint(self) -> None:
        self._check(r"""
for(const compact of [false,true,false,true]) {
  mediaStub.setMatches(compact);
  for(const id of ['filter-disclosure','legend-disclosure'])assert.equal(elements[id].open,!compact);
  render();
}
assert.equal(mediaStub.listeners.length,1);
""", compact=True)

    def test_mobile_user_disclosure_choice_survives_graph_renders(self) -> None:
        self._check(r"""
for(const id of ['filter-disclosure','legend-disclosure'])elements[id].open=true;
mediaStub.setMatches(true);
render();
search.value=targets.support; search.fire('input');
view.value='full'; view.fire('change');
for(const id of ['filter-disclosure','legend-disclosure'])assert.equal(elements[id].open,true);
elements['legend-disclosure'].open=false;
render();
assert.equal(elements['filter-disclosure'].open,true);
assert.equal(elements['legend-disclosure'].open,false);
""", compact=True)

    def test_argument_mode_hides_filter_wrapper_and_preserves_view_access(self) -> None:
        self._check(r"""
const filter=elements['filter-disclosure'];
assert.ok(graphControls.includes(filter));
filter.open=true;
view.value='argument'; view.fire('change');
assert.equal(filter.hidden,true);
assert.equal(view.hidden,false);
assert.equal(elements['legend-disclosure'].hidden,false);
assert.equal(argumentView.hidden,false);
view.value='overview'; view.fire('change');
assert.equal(filter.hidden,false);
assert.equal(filter.open,true);
assert.equal(view.hidden,false);
assert.equal(sourceIntegrity.hidden,false);
assert.equal(graphMain.hidden,false);
""", compact=True)

    def test_argument_support_review_and_gate_jump_to_visible_details(self) -> None:
        self._check(r"""
for(const role of ['support','review','gate']) {
  view.value='argument'; render();
  assert.equal(graphMain.hidden,true);
  sourceControl.value='research_vcs'; layer.value='wrong'; kind.value='wrong'; proof.value='wrong'; search.value='old query';
  const button=registeredClaims.querySelectorAll('.node-jump').find(el=>el.dataset.nodeId===targets[role]);
  assert.ok(button,role+' button must exist');
  button.fire('click');
  assert.equal(view.value,'overview',role+' must leave Argument');
  assert.equal(graphMain.hidden,false);
  assert.equal(argumentView.hidden,true);
  assert.equal(sourceIntegrity.hidden,false);
  assert.ok(graphControls.every(el=>!el.hidden));
  assert.equal(selectedId,targets[role]);
  assert.equal(search.value,targets[role]);
  assert.equal(sourceControl.value,'all');
  for(const control of [layer,kind,proof]) assert.equal(control.value,'');
  assert.ok(details.innerHTML.includes(targets[role]));
  const expected=[...oneHopIds(new Set([targets[role]]),allEdges)].sort();
  assert.deepEqual([...currentDisplayById.keys()].sort(),expected);
}
""")

    def test_argument_rerenders_bind_each_new_button_once(self) -> None:
        self._check(r"""
for(let iteration=0;iteration<3;iteration++) {
  view.value='argument'; render(); render();
  const buttons=registeredClaims.querySelectorAll('.node-jump');
  assert.ok(buttons.length>0);
  for(const button of buttons) assert.equal((button.events.click||[]).length,1);
  const writes=elements.canvas.writes;
  buttons.find(button=>button.dataset.nodeId===targets.support).fire('click');
  assert.equal(elements.canvas.writes,writes+1);
  assert.equal(selectedId,targets.support);
}
""")

    def test_detail_jumps_preserve_graph_mode_and_exact_one_hop_search(self) -> None:
        self._check(r"""
for(const mode of ['overview','full']) {
  view.value=mode; render();
  details.innerHTML=idList([targets.review]); wireDetailJumps();
  const button=details.querySelectorAll('.node-jump')[0];
  assert.equal(button.events.click.length,1);
  button.fire('click');
  assert.equal(view.value,mode);
  assert.equal(search.value,targets.review);
  assert.equal(selectedId,targets.review);
  assert.ok(details.innerHTML.includes(targets.review));
  assert.deepEqual([...currentDisplayById.keys()].sort(),[targets.review,targets.gate].sort());
}
""")

    def test_unknown_target_is_plain_text_and_focus_is_noop(self) -> None:
        self._check(r"""
const missing='rso-not-in-graph';
assert.equal(nodeRef(missing),'<code>'+missing+'</code>');
for(const mode of ['argument','full']) {
  view.value=mode; render();
  const before=[view.value,search.value,selectedId,sourceControl.value,layer.value,kind.value,proof.value,elements.canvas.writes,details.innerHTML];
  focusNode(missing);
  assert.deepEqual([view.value,search.value,selectedId,sourceControl.value,layer.value,kind.value,proof.value,elements.canvas.writes,details.innerHTML],before);
}
""")
