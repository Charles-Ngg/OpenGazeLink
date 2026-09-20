/* Serializable trials. The future stimulus schedule is telemetry, never an inference input. */
(() => {
  function plan(seed=9409) {
    let state=seed>>>0;
    const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;};
    const steps=[];
    for(const [split,count,group] of [["train",24,"动态适配"],["validation",12,"检查预测"],["test",12,"最终验证"]]) {
      for(let i=0;i<count;i++) {
        const profile=["line","accelerate","stop_reverse","arc","jump","mixed"][i%6];
        const angle=(i/6)*Math.PI/2+(split==="train"?0:split==="validation"?.31:.67);
        const extent=.25+random()*.14;
        const a=[.5-Math.cos(angle)*extent,.5-Math.sin(angle)*extent];
        const b=[1-a[0],1-a[1]];
        const points=Array.from({length:4},()=>[.12+.76*random(),.12+.76*random()]);
        const changes=[900+random()*300,2400+random()*300,4000+random()*300];
        steps.push({phase:"pursuit",point:a,end:b,duration:6000,block:`${split}-prediction-${i}`,
          trial_id:`${split}-prediction-${i}`,split,group,condition:"normal",motion_profile:profile,
          angle,extent,points,changes,motion_duration_ms:2600+random()*1000,seed});
      }
    }
    return steps;
  }
  function sample(step,age) {
    const t=Math.max(0,Math.min(step.duration,age));
    let phase="pursuit",p=0,position;
    const u=Math.max(0,Math.min(1,(t-900)/step.motion_duration_ms));
    if(step.motion_profile==="jump" || step.motion_profile==="mixed") {
      const k=step.changes.filter(c=>t>=c).length;
      position=[...step.points[k]];
      phase="jump";
      if(step.motion_profile==="mixed") {
        const since=t-(k?step.changes[k-1]:0);
        position[0]+=.045*Math.sin(since/600);
        position[1]+=.035*Math.sin(since/850);
        phase="pursuit";
      }
    } else if(step.motion_profile==="arc") {
      const theta=step.angle+u*Math.PI*1.7;
      position=[.5+step.extent*Math.cos(theta),.5+.7*step.extent*Math.sin(theta)];
    } else {
      p=step.motion_profile==="accelerate" ? u*u*(3-2*u) : u;
      if(step.motion_profile==="stop_reverse") {
        // Travel, stop for 600 ms, reverse, then stop again.
        p=t<900?0:t<2300?(t-900)/1400:t<2900?1:t<4600?1-(t-2900)/1700:0;
      }
      position=step.point.map((a,i)=>a+p*(step.end[i]-a));
    }
    if(!["jump","mixed"].includes(step.motion_profile)) {
      if(t<900 || (step.motion_profile==="stop_reverse" ? (t>=2300&&t<2900)||t>=4600 : u>=1))phase="anchor";
    }
    return {position,phase};
  }
  const api={plan,sample};
  if(typeof module!=="undefined")module.exports=api;else window.PredictionPlan=api;
})();
