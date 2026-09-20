const assert=require('assert');
const {spatialPlan,dragProgress,eventPlan,headCue}=require('../web/unified-plan');
for(let seed=0;seed<100;seed++) {
  const plan=spatialPlan(seed),events=eventPlan(seed);
  assert.equal(events.length,12);assert(events.reduce((n,s)=>n+s.duration,0)<72000);
  for(const split of ['train','validation','test'])assert(events.filter(s=>s.split===split).length>=3);
  assert.equal(plan.length,12);
  assert.deepStrictEqual(plan,spatialPlan(seed));
  assert.equal(new Set(plan.map(s=>s.trial_id)).size,12);
  assert(!plan.some(s=>s.split==='validation'));
  for(const split of ['train','test']) {
    const local=plan.filter(s=>s.split===split);
    assert.equal(local.length,split==='train'?8:4);
    assert(local.every(s=>s.motion_profile==='line'));
  }
  assert.equal(new Set(plan.map(s=>JSON.stringify([s.point,s.end].sort()))).size,12);
  for(const step of plan) {
    assert.equal(step.calibration_stage,'spatial_v1');
    assert.equal(step.plan_version,10);assert.equal(step.sample_rate_hz,120);assert(!step.head_pose);assert(!step.head_mode);
    assert(step.point.every(v=>v>=0&&v<=1));
    const cues=[headCue(step,0,'ready'),headCue(step,.1,'settling'),headCue(step,.5,'moving'),headCue(step,.9,'landing')];
    assert(cues.every(c=>!/头|Head|head/.test(c.text)&&c.condition==='normal'));
    if(step.motion_profile!=='line')continue;
    const length=Math.hypot(...step.point.map((v,i)=>v-step.end[i]));
    assert(length>=.86);
    const dx=Math.abs(step.end[0]-step.point[0]),dy=Math.abs(step.end[1]-step.point[1]);
    assert((dx>=.86&&dy<=.03)||(dy>=.86&&dx<=.03)||(dx>=.86&&dy>=.86));
    for(const fps of [30,60,120,144,240]) {
      let p=0;
      for(let i=0;i<fps*10;i++) {
        const next=dragProgress(step,p,1,1000/fps);
        assert(next>=p&&next<=1);
        assert((next-p)*length<=step.max_speed/fps+1e-9);
        assert.equal(dragProgress(step,next,0,1000/fps),next);
        p=next;
      }
      assert(p>=.999);
    }
  }
}
console.log('100 seeds: 12 unique rails, train/test only, no head prompts, 120 Hz spatial sampling ceiling, bounded response.');
