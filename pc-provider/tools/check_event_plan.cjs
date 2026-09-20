const assert=require('assert');
const {eventPlan,sample}=require('../web/unified-plan');
for(let seed=0;seed<100;seed++) {
 for(const extended of [false,true]) {
  const plan=eventPlan(seed,{extended});
  assert.deepStrictEqual(plan,eventPlan(seed,{extended}));
  assert.equal(new Set(plan.map(s=>s.trial_id)).size,extended?48:12);
  if(!extended) {
    assert(plan.reduce((sum,s)=>sum+s.duration,0)<72000);
    for(const band of ['short','medium','long'])for(let axis=0;axis<4;axis++)
      assert.equal(plan.filter(s=>s.amplitude_band===band&&s.direction_axis===axis).length,1);
  }
  for(const [split,repeats] of [['train',2],['validation',1],['test',1]]) {
    const local=plan.filter(s=>s.split===split);
    assert.equal(local.length,repeats*(extended?12:3));
    assert.equal(new Set(local.map(s=>s.direction_axis)).size,extended||split==='train'?4:3);
    for(const band of ['short','medium','long']) {
      assert.equal(local.filter(s=>s.amplitude_band===band).length,repeats*(extended?4:1));
      if(extended)for(let axis=0;axis<4;axis++)
        assert.equal(local.filter(s=>s.amplitude_band===band&&s.direction_axis===axis).length,repeats);
    }
  }
  for(const step of plan) {
    const limits={short:[.06,.10],medium:[.18,.24],long:[.38,.48]}[step.amplitude_band];
    assert(step.points.flat().every(v=>v>=.02&&v<=.98));
    for(let k=1;k<3;k++) {
      const d=Math.hypot(...step.points[k].map((v,i)=>v-step.points[k-1][i]));
      assert(d>=limits[0]-1e-8&&d<=limits[1]+1e-8);
      assert.deepStrictEqual(sample(step,step.changes[k-1]-.01).position,step.points[k-1]);
      assert.deepStrictEqual(sample(step,step.changes[k-1]).position,step.points[k]);
    }
    const bounds=[0,...step.changes,step.duration];
    assert(bounds.slice(1).every((v,i)=>v-bounds[i]>=1100));
  }
 }
}
console.log('100 seeds each: v9 12 trials / 6-3-3 splits / under 72s; v8 48 trials retained; balanced distances, direction coverage, varied holds, exact jump boundaries.');
