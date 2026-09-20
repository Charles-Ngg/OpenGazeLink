import unittest
import numpy as np
import torch
from opengazelink_pc.unified_prediction import MotionStateNetwork, UnifiedHistory, CONTEXT_DIM, UnifiedNetwork, RefinedMotionNetwork, ResidualTrajectoryNetwork
from opengazelink_pc.unified_prediction_training import preserves_dynamic_accuracy


class MotionStateTests(unittest.TestCase):
    def test_stability_tradeoff_cannot_mask_motion_regression(self):
        import copy
        baseline={'mean_proxy_px':100,'p95_proxy_px':300,'groups':{
            'pursuit':{'mean_px':120,'p95_px':350},'saccade_proxy':{'mean_px':200,'p95_px':500}}}
        candidate=copy.deepcopy(baseline);candidate['mean_proxy_px']=95
        self.assertTrue(preserves_dynamic_accuracy(candidate,baseline))
        candidate['groups']['pursuit']['mean_px']=121
        self.assertFalse(preserves_dynamic_accuracy(candidate,baseline))

    def test_frozen_incumbent_and_zero_motion_export_in_refinements(self):
        for cls in (RefinedMotionNetwork,ResidualTrajectoryNetwork):
            incumbent=UnifiedNetwork(np.zeros(CONTEXT_DIM),np.ones(CONTEXT_DIM),False,False,0.)
            model=cls(incumbent,np.zeros(CONTEXT_DIM),np.ones(CONTEXT_DIM)).eval()
            self.assertTrue(all(not p.requires_grad for p in model.incumbent.parameters()))
            context=torch.zeros(3,CONTEXT_DIM);experts=torch.zeros(3,6,4)
            context[:,64:148].reshape(3,12,7)[:,:,6]=1
            horizon=torch.tensor([[.033],[.085],[.25]])
            traced=torch.jit.trace(model,(context[:1],experts[:1],horizon[:1]),strict=False)
            for a,b in zip(model(context,experts,horizon),traced(context,experts,horizon)):
                torch.testing.assert_close(a,b)
            torch.testing.assert_close(traced(context,experts,horizon)[0],torch.zeros(3,2))

    def test_stationary_history_has_no_invented_motion_at_any_horizon(self):
        model = MotionStateNetwork(np.zeros(CONTEXT_DIM),np.ones(CONTEXT_DIM)).eval()
        history=UnifiedHistory()
        for time in [0,23,57,91,130,165,203,241,276,310,343,380]:
            context,experts,_=history.update(time,[.4,.6],[.4,.6],np.ones(404),np.zeros(64))
        c=torch.tensor(np.repeat(context[None],5,0))
        e=torch.tensor(np.repeat(experts[None],5,0))
        h=torch.tensor([[0.],[.033],[.085],[.125],[.25]])
        torch.testing.assert_close(model(c,e,h)[0],torch.zeros(5,2),atol=1e-6,rtol=0)

    def test_irregular_constant_velocity_export_and_prefix_match(self):
        model=MotionStateNetwork(np.zeros(CONTEXT_DIM),np.ones(CONTEXT_DIM)).eval()
        history=UnifiedHistory();contexts=[];experts=[]
        for time in [0,23,57,91,130,165,203,241,276,310,343,380]:
            p=np.array([.3,.4])+time*np.array([.0002,-.0001])
            c,e,_=history.update(time,p,p,np.zeros(404),np.zeros(64))
            contexts.append(c);experts.append(e)
        c,e=torch.tensor(np.array(contexts)),torch.tensor(np.array(experts))
        h=torch.linspace(.016,.25,len(c))[:,None]
        state=model.state(c[-1:])
        torch.testing.assert_close(state[2],torch.tensor([[.2,-.1]]),atol=1e-5,rtol=0)
        torch.testing.assert_close(state[3],torch.zeros(1,2),atol=1e-4,rtol=0)
        traced=torch.jit.trace(model,(c[:1],e[:1],h[:1]),strict=False)
        batch=traced(c,e,h)
        for i in range(len(c)):
            online=traced(c[i:i+1],e[i:i+1],h[i:i+1])
            for a,b in zip(batch,online):torch.testing.assert_close(a[i:i+1],b,atol=1e-6,rtol=1e-5)
        for a,b in zip(model(c,e,h),batch):torch.testing.assert_close(a,b)
        # Hidden state and external filter cannot silently change intrinsic state.
        changed=c.clone();changed[:,:64]+=100
        altered=e.clone();altered[:,1]+=100
        for a,b in zip(batch,traced(changed,altered,h)):torch.testing.assert_close(a,b)


if __name__=='__main__':unittest.main()
