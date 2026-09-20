const assert=require('assert');
const {plan,sample}=require('../web/unified-plan');
const reports=[];
for(let seed=0;seed<100;seed++) {
 const steps=plan(seed),ids=new Set(steps.map(s=>s.trial_id));
 assert.equal(steps.length,72);assert.equal(ids.size,72);
 assert.deepStrictEqual(steps,plan(seed));
 const distance=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
 const anchors=steps.filter(s=>s.motion_profile==='anchor');
 for(let i=0;i<anchors.length;i++)for(let j=0;j<i;j++)assert(distance(anchors[i].point,anchors[j].point)>=.065);
 for(let i=0;i<steps.length;i++)for(let j=0;j<i;j++) {
  if(steps[i].motion_profile!=='anchor'&&steps[i].motion_profile===steps[j].motion_profile)
   assert(distance(steps[i].center,steps[j].center)>=.14);
 }
 const summary={seed,durationSeconds:steps.reduce((n,s)=>n+s.duration,0)/1000,splits:{}};
 assert.equal(summary.durationSeconds,237.6);
 for(const split of ['train','validation','test']) {
  const local=steps.filter(s=>s.split===split),anchors=local.filter(s=>s.motion_profile==='anchor');
  const cells=new Set(anchors.map(s=>s.point.map(x=>Math.min(2,Math.floor(x*3))).join(',')));
  assert.equal(cells.size,9);
  const profiles={};for(const s of local)profiles[s.motion_profile]=(profiles[s.motion_profile]||0)+1;
  assert.equal(Object.keys(profiles).length,13);summary.splits[split]=profiles;
  for(let b=0;b<3;b++)assert.equal(local.filter(s=>s.rest_block===b).length,split==='train'?10:7);
 }
 for(const step of steps)for(let age=0;age<=step.duration;age+=17) {
  const value=sample(step,age);assert(value.position.every(x=>Number.isFinite(x)&&x>=.02&&x<=.98));
  if(value.rail) {
   assert.equal(value.phase,'pursuit');
   const [a,b]=[value.rail.a,value.rail.b];
   assert(Math.abs((value.position[0]-a[0])*(b[1]-a[1])-(value.position[1]-a[1])*(b[0]-a[0]))<1e-8);
  }
  if(step.motion_profile!=='anchor'&&step.motion_profile!=='jump'&&step.motion_profile!=='mixed'&&age>0) {
   // Continuous profiles must not teleport at holds, reversals or corners.
   assert(distance(value.position,sample(step,age-1).position)<.003);
  }
 }
 for(const step of steps.filter(s=>!['anchor','jump','mixed'].includes(s.motion_profile))) {
  assert.equal(sample(step,0).phase,'anchor');assert.equal(sample(step,step.duration).phase,'anchor');
 }
 if(seed===0)reports.push(summary);
}
console.log(JSON.stringify({testedSeeds:100,reports}));
console.log('72 trials / 237.6 seconds; 12 motion profiles per split; separated anchors and same-profile centers; bounded continuous paths and correct rails.');
