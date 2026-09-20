const assert=require('assert');
const {plan,sample}=require('../web/prediction-plan.js');
const trials=plan(9410);
assert.deepStrictEqual(trials,plan(9410));
assert.notDeepStrictEqual(trials,plan(9411));
assert.equal(new Set(trials.map(t=>t.trial_id)).size,trials.length);
assert.equal(trials.reduce((sum,t)=>sum+t.duration,0),288000);
for(const split of ['train','validation','test']) {
  const group=trials.filter(t=>t.split===split);
  assert.equal(new Set(group.map(t=>t.motion_profile)).size,6);
  for(const t of group) {
    assert(t.block.startsWith(split+'-'));
    for(let ms=0;ms<=6000;ms+=16) {
      const s=sample(t,ms);
      assert(s.position.every(x=>Number.isFinite(x)&&x>=0&&x<=1));
      assert(['anchor','pursuit','jump'].includes(s.phase));
    }
    if(t.motion_profile==='stop_reverse') {
      assert.deepStrictEqual(sample(t,2400).position,sample(t,2800).position);
      assert.deepStrictEqual(sample(t,0).position,sample(t,5000).position);
    }
    if(t.motion_profile==='jump') {
      assert.notDeepStrictEqual(sample(t,t.changes[0]-1).position,sample(t,t.changes[0]+1).position);
    }
  }
}
console.log(JSON.stringify({trials:trials.length,duration_seconds:288,profiles_per_split:6,result:'passed'}));
