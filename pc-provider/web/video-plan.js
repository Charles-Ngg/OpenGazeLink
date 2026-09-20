/* Pure calibration geometry, shared by the page and coverage verification. */
(() => {
  function plan() {
    const steps = [];
    const anchor = (point, block, group, condition="normal") => steps.push({
      phase:"anchor", point, duration:1150, block, group, condition,
    });
    const rail = (a,b,block,group,condition="normal",speed=.16) => {
      anchor(a,`${block}-start`,group,condition);
      steps.push({phase:"pursuit",point:a,end:b,block,group,condition,speed});
      anchor(b,`${block}-end`,group,condition);
    };
    [[.5,.5],[.06,.06],[.94,.94],[.94,.06],[.06,.94]].forEach((p,i)=>anchor(p,`train-intro-${i}`,"准备"));
    const rails = [];
    [.06,.5,.94].forEach(y=>rails.push([[.06,y],[.94,y]]));
    [.06,.5,.94].forEach(x=>rails.push([[x,.94],[x,.06]]));
    rails.forEach(([a,b],i)=>rail(a,b,`train-forward-${i}`,"覆盖屏幕"));
    rails.forEach(([a,b],i)=>rail(b,a,`train-reverse-${i}`,"反向与适应",
      ["normal","head_left","dim","head_right","bright","normal"][i],i%2 ? .14 : .16));
    // Whole blocks for selection, followed by an untouched final evaluation.
    [[.5,.5],[.08,.08],[.92,.92],[.92,.08],[.08,.92]].forEach((p,i)=>anchor(p,`validation-point-${i}`,"检查适配"));
    rail([.08,.3],[.92,.3],"validation-h","检查适配");
    rail([.7,.92],[.7,.08],"validation-v","检查适配");
    rail([.08,.78],[.92,.78],"validation-h2","检查适配");
    [[.5,.5],[.2,.15],[.8,.85],[.8,.15],[.2,.85]].forEach((p,i)=>anchor(p,`test-point-${i}`,"最终验证"));
    rail([.92,.4],[.08,.4],"test-h","最终验证", "normal",.14);
    rail([.35,.08],[.35,.92],"test-v","最终验证", "normal",.14);
    return steps;
  }
  function project(pointer,a,b,width,height) {
    const dx=(b[0]-a[0])*width, dy=(b[1]-a[1])*height;
    const progress=Math.max(0,Math.min(1,((pointer[0]-a[0])*width*dx+(pointer[1]-a[1])*height*dy)/(dx*dx+dy*dy)));
    return (1-progress)*Math.hypot(dx,dy)<6 ? 1 : progress;
  }
  function advance(progress,desired,velocity,dt,length,maxSpeed) {
    // Mouse controls requested distance. Motion remains forward and bounded;
    // reverse traversals are explicit trials with their own recorded identity.
    const remaining=Math.max(0,desired-progress)*length;
    const wanted=Math.min(maxSpeed,Math.sqrt(2*maxSpeed*3*remaining));
    const dv=maxSpeed*3*dt;
    const nextVelocity=Math.max(0,Math.min(velocity+dv,Math.max(velocity-dv,wanted)));
    let next=Math.min(1,Math.max(progress,Math.min(desired,progress+nextVelocity*dt/length)));
    if(desired===1 && (1-next)*length<.5)next=1;
    return {progress:next,velocity:next===progress?0:nextVelocity};
  }
  const api={plan,project,advance};
  if(typeof module!=="undefined")module.exports=api;
  else window.VideoRailPlan=api;
})();
