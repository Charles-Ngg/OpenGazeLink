/* One spatially diverse schedule supplies calibration and causal prediction. */
(() => {
  const profiles=['line','accelerate','decelerate','stop_reverse','arc','wave','zigzag','loop','spiral','corner','jump','mixed'];
  const distance=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
  const lerp=(a,b,u)=>a.map((v,i)=>v+(b[i]-v)*u);
  const clamp=x=>Math.max(0,Math.min(1,x));
  function transform(step,p) {
    const c=Math.cos(step.angle),s=Math.sin(step.angle),norm=Math.abs(c)+Math.abs(s);
    return [step.center[0]+step.radius[0]*(p[0]*c-p[1]*s)/norm,
      step.center[1]+step.radius[1]*(p[0]*s+p[1]*c)/norm];
  }
  function plan(seed=9410) {
    let state=seed>>>0;
    const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;};
    const shuffle=a=>{for(let i=a.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;};
    const anchors=[],motions=[],usedAnchors=[];
    const ranges=[[.055,.245],[.38,.62],[.755,.945]];
    // Spread all splits across nine cells without repeating coordinates.
    for(let y=0;y<3;y++)for(let x=0;x<3;x++) {
      for(const split of shuffle(['train','train','validation','test'])) {
        let point,best=-1;
        for(let k=0;k<160;k++) {
          const p=[x,y].map(cell=>ranges[cell][0]+random()*(ranges[cell][1]-ranges[cell][0]));
          const separation=Math.min(1,...usedAnchors.map(q=>distance(p,q)));
          if(separation>best){best=separation;point=p;}
        }
        if(best<.065)throw new Error('注视点间距不足，请重新准备采集');
        usedAnchors.push(point);
        const id=`${split}-anchor-${anchors.filter(s=>s.split===split).length}`;
        anchors.push({phase:'anchor',motion_profile:'anchor',point,duration:1800,block:id,trial_id:id,split,condition:'normal'});
      }
    }
    for(const [profileIndex,profile] of profiles.entries()) {
      const centers=[];
      for(const [splitIndex,split] of ['train','validation','test'].entries()) {
        let center,best=-1;
        for(let k=0;k<80;k++) {
          const p=[.3+random()*.4,.3+random()*.4];
          const separation=Math.min(1,...centers.map(q=>distance(p,q)));
          if(separation>best){best=separation;center=p;}
        }
        if(best<.14)throw new Error('同类轨迹间距不足，请重新准备采集');
        centers.push(center);
        const id=`${split}-motion-${profileIndex}`;
        const step={phase:'pursuit',motion_profile:profile,center,
          angle:(profileIndex%4)*Math.PI/4+splitIndex*Math.PI*2/3+(random()-.5)*.35,
          radius:center.map(v=>Math.min(v-.045,.955-v)*(.8+random()*.18)),
          direction:random()<.5?-1:1,motion_duration_ms:2400+random()*500,
          changes:profile==='mixed'?[2300+random()*200]:[1400+random()*150,3000+random()*150],
          duration:4800,block:id,trial_id:id,split,condition:'normal'};
        step.points=shuffle([[-.9,-.75],[.85,-.6],[.1,.95]]).map(p=>transform(step,p));
        step.point=transform(step,[-1,0]);step.end=transform(step,[1,0]);
        motions.push(step);
      }
    }
    shuffle(anchors);shuffle(motions);
    const steps=[];
    // Three balanced groups; avoid presenting the same pattern consecutively.
    for(let block=0;block<3;block++) {
      const local=[];
      for(const [split,ac] of [['train',6],['validation',3],['test',3]]) {
        for(const [source,count] of [[anchors,ac],[motions,4]])for(let n=0;n<count;n++) {
          const i=source.findIndex(s=>s.split===split);local.push(source.splice(i,1)[0]);
        }
      }
      shuffle(local);
      while(local.length) {
        const previous=steps.at(-1)?.motion_profile;
        const anchorCount=local.filter(s=>s.motion_profile==='anchor').length;
        let i=previous!=='anchor'&&anchorCount>=local.length-anchorCount
          ?local.findIndex(s=>s.motion_profile==='anchor')
          :local.findIndex(s=>s.motion_profile!==previous);
        if(i<0)i=0;
        steps.push({...local.splice(i,1)[0],rest_block:block,group:`第 ${block+1}/3 组`,seed,plan_version:2});
      }
    }
    return steps;
  }
  function sample(step,age) {
    if(step.motion_profile==='anchor')return {position:[...step.point],phase:'anchor',rail:null,motionAge:0};
    const t=Math.max(0,Math.min(step.duration,age)),u=clamp((t-900)/step.motion_duration_ms);
    const profile=step.motion_profile;
    let phase=t<900||u>=1?'anchor':'pursuit',position,rail=null,motionAge=t-900;
    if(profile==='jump'||profile==='mixed') {
      const k=step.changes.filter(c=>t>=c).length,since=t-(k?step.changes[k-1]:0);
      position=[...step.points[k]];phase='jump';
      if(profile==='mixed') {
        const p=clamp((since-850)/650);
        position=lerp(position,lerp(position,step.center,.5),p);
        phase=p>0&&p<1?'pursuit':'anchor';motionAge=since-850;
      }
    } else if(['line','accelerate','decelerate','stop_reverse'].includes(profile)) {
      let p=profile==='accelerate'?u*u:profile==='decelerate'?1-(1-u)**2:u;
      if(profile==='stop_reverse') {
        p=t<900?0:t<2000?(t-900)/1100:t<2700?1:t<3800?1-(t-2700)/1100:0;
        phase=t<900||(t>=2000&&t<2700)||t>=3800?'anchor':'pursuit';
        motionAge=t<2000?t-900:t-2700;
      }
      position=lerp(step.point,step.end,p);
      if(phase==='pursuit')rail={a:step.point,b:step.end};
    } else if(profile==='zigzag'||profile==='corner') {
      const path=(profile==='zigzag'?[[-1,-.6],[-.35,.7],[.35,-.7],[1,.6]]:[[-1,-.8],[-1,.8],[1,.8]])
        .map(p=>transform(step,p));
      const leg=Math.min(path.length-2,Math.floor(u*(path.length-1))),p=u*(path.length-1)-leg;
      position=lerp(path[leg],path[leg+1],p);
      motionAge=(u-leg/(path.length-1))*step.motion_duration_ms;
      if(phase==='pursuit')rail={a:path[leg],b:path[leg+1]};
    } else {
      let p;
      const theta=step.direction*u*Math.PI*(profile==='arc'?1.4:profile==='spiral'?2.5:2);
      if(profile==='wave')p=[2*u-1,.65*Math.sin(step.direction*u*2*Math.PI)];
      else {const radius=profile==='spiral'?.25+.75*u:1;p=[radius*Math.cos(theta),radius*Math.sin(theta)];}
      position=transform(step,p);
    }
    return {position,phase,rail,motionAge:Math.max(0,motionAge)};
  }
  function spatialPlan(seed=9410,{width=1920,height=1080}={}) {
    if(!Number.isFinite(width)||!Number.isFinite(height)||width<320||height<240)throw new Error('Invalid capture viewport');
    let state=seed>>>0;
    const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;};
    const shuffle=a=>{for(let i=a.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;};
    // Only the 17px target radius plus 1px remains outside the sampled area.
    // Four perimeter rails reach every corner twice; four inner rails retain
    // central training coverage. Model-selection rails are separate trials.
    const x=18/width,y=18/height;
    const geometries={
      train:[
        [[x,y],[1-x,y]], [[1-x,1-y],[x,1-y]],
        [[x,1-y],[x,y]], [[1-x,y],[1-x,1-y]],
        [[x,.4],[1-x,.4]], [[1-x,.6],[x,.6]],
        [[.4,1-y],[.4,y]], [[.6,y],[.6,1-y]],
      ],
      test:[
        [[2*x,2*y],[1-2*x,1-2*y]], [[1-2*x,2*y],[2*x,1-2*y]],
        [[x,.5],[1-x,.5]], [[.5,1-y],[.5,y]],
      ],
    };
    const lines=[];
    for(const split of ['train','test'])for(const [index,[point,end]] of geometries[split].entries()) {
      const id=`${split}-spatial-${index}`;
      lines.push({trial_id:id,block:id,split,motion_profile:'line',point,end,condition:'normal',
        duration:5200,calibration_stage:'spatial_v1',plan_version:10,seed,
        viewport_width:width,viewport_height:height,edge_margin_px:18,
        sample_rate_hz:120,spatial_balance:'equal_3x3_regions',
        response_ms:180,max_speed:distance(point,end)/4.2,settle_ms:1000,end_hold_ms:1000});
    }
    return shuffle(lines).map((step,index)=>({...step,rest_block:Math.floor(index/4),group:`第 ${Math.floor(index/4)+1}/3 组`}));
  }
  function headCue(step,progress=0,state='ready',language='zh') {
    // Keep the telemetry helper's API compatible, without prescribing head motion.
    const text=state==='ready'?(language==='zh'?'注视蓝点并沿线拖动':'Look at the blue dot and drag along the line'):
      (language==='zh'?'眼睛跟随蓝点':'Follow the blue dot with your eyes');
    return {condition:'normal',text};
  }
  function eventPlan(seed=9410,{extended=false}={}) {
    let state=seed>>>0;
    const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;};
    const steps=[];
    // Distance is Euclidean normalized-screen distance, NOT visual degrees.
    // V9: cross distance x axis once overall, split whole trials 6/3/3.
    // Each split contains all distances; per-split axis coverage is limited.
    // V8 remains available for reproducible analysis of older recordings.
    // Long stationary holds also supply negative examples for false-onset tests.
    const bands=[['short',.06,.10],['medium',.18,.24],['long',.38,.48]];
    const offset=seed>>>0;
    for(const [split,repeats] of [['train',extended?2:1],['validation',1],['test',1]])
      for(let repeat=0;repeat<repeats;repeat++)for(const [bandIndex,[band,lo,hi]] of bands.entries())for(let axis=0;axis<4;axis++) {
        const slot=(axis+bandIndex+offset%4)%4;
        if(!extended && split!==['train','train','validation','test'][slot])continue;
        const id=`${split}-event-${repeat}-${band}-${axis}`;
        const angle=axis*Math.PI/4+(random()-.5)*.16+(random()<.5?Math.PI:0);
        const amplitude=lo+random()*(hi-lo),delta=[Math.cos(angle),Math.sin(angle)].map(v=>v*amplitude/2);
        const center=delta.map(v=>{const margin=.07+Math.abs(v);return margin+random()*(1-2*margin);});
        const a=center.map((v,k)=>v-delta[k]),b=center.map((v,k)=>v+delta[k]);
        const holds=[1100+Math.floor(random()*600),1400+Math.floor(random()*600),1400+Math.floor(random()*600)];
        steps.push({trial_id:id,block:id,split,condition:'normal',calibration_stage:'events_v1',
          motion_profile:'jump',points:[a,b,a],point:a,amplitude_band:band,amplitude_normalized:amplitude,
          direction_axis:axis,changes:[holds[0],holds[0]+holds[1]],duration:holds.reduce((a,b)=>a+b),
          settle_guard_ms:700,reaction_window_ms:[80,600],distance_unit:'normalized_screen',
          rest_block:0,group:'分距离眼跳与注视',seed,plan_version:extended?8:9});
      }
    // Shuffle whole trials; held-out event sequences are never training frames.
    for(let i=steps.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[steps[i],steps[j]]=[steps[j],steps[i]];}
    return steps;
  }
  function dragProgress(step,current,desired,dt) {
    const length=distance(step.point,step.end);
    // Monotonic progress prevents accidental backtracking or corners.
    const goal=Math.max(current,clamp(desired));
    return Math.min(1,current+Math.min(goal-current,(goal-current)*(1-Math.exp(-dt/step.response_ms)),
      step.max_speed*dt/1000/length));
  }
  const api={plan,sample,spatialPlan,dragProgress,headCue,eventPlan};
  if(typeof module!=='undefined')module.exports=api;else window.UnifiedPlan=api;
})();
