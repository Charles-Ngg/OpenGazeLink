/* Static/VM contract checks. Actual browser QA uses preview_control_ui.py. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..', 'web');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
assert.equal(ids.length, new Set(ids).size, 'HTML ids must be unique');
const known = new Set(ids);
for (const name of ['app.js', 'video.js']) {
  const source = fs.readFileSync(path.join(root, name), 'utf8');
  for (const match of source.matchAll(/\$\(["']([^"']+)["']\)/g)) {
    assert(known.has(match[1]), `${name} refers to missing #${match[1]}`);
  }
  new vm.Script(source, {filename: name});
}
for (const removed of ['gazeModel', 'landmarker', 'lightingProfile', 'oneEuroEnabled',
                       'videoForecastEnabled', 'extrapolationEnabled', 'startPredictionButton', 'modelsGrid']) {
  assert(!known.has(removed), `Obsolete control #${removed} returned`);
}
assert.equal([...html.matchAll(/role="switch"/g)].length, 1);
for (const required of ['eventTemporalEnabled', 'startVideoButton', 'startEventButton', 'language',
                        'latencyChart', 'latencyStageRows', 'latencyEndToEnd']) assert(known.has(required));
const scripts = [...html.matchAll(/<script src="([^"]+)"/g)].map(m => m[1]);
assert.deepEqual(scripts, ['/i18n.js', '/app.js', '/unified-plan.js', '/video.js']);
for (const tag of html.matchAll(/<[^>]+data-(?:zh|en)=[^>]+>/g)) {
  assert.match(tag[0], /data-zh="[^"]+"/);
  assert.match(tag[0], /data-en="[^"]+"/);
}
const model = {window:{}};
vm.createContext(model);
vm.runInContext(fs.readFileSync(path.join(root, 'unified-plan.js'), 'utf8'), model);
for (let seed = 0; seed < 100; seed++) {
  for (const step of [...model.window.UnifiedPlan.spatialPlan(seed), ...model.window.UnifiedPlan.eventPlan(seed)]) {
    const zh = model.window.UnifiedPlan.headCue(step, .5, 'moving', 'zh');
    const en = model.window.UnifiedPlan.headCue(step, .5, 'moving', 'en');
    assert.equal(en.condition, zh.condition, 'UI language cannot alter supervision labels');
    assert(en.text && !/[\u4e00-\u9fff]/.test(en.text), 'English cue contains untranslated text');
  }
}
console.log('Control UI contract: unique ids, all JS targets, one prediction switch, two stages, performance, bilingual cues (100 seeds).');
