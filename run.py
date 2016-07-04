#!/usr/bin/env python
"""
This script runs a policy gradient algorithm
"""


import gym, logging
import argparse, sys, pickle 
from tabulate import tabulate
import shutil, os, logging

from rivuletpy.rivuletenv import RivuletEnv, RandomAgent 
from rivuletpy.lib.modular_rl import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)    
    parser.add_argument("--agent",required=True)
    parser.add_argument("--plot",action="store_true")
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    env = RivuletEnv(imgpath='tests/data/test-small.tif', swcpath='tests/data/test-small.swc', cached=False, nsonar=60)
    # env_spec = env.spec
    mondir = args.outfile + ".dir"
    if os.path.exists(mondir): shutil.rmtree(mondir)
    os.mkdir(mondir)
    env.monitor.start(mondir, video_callable=None if args.video else VIDEO_NEVER)
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    # if args.timestep_limit == 0: 
    #     args.timestep_limit = env_spec.timestep_limit    
    cfg = args.__dict__
    np.random.seed(args.seed)
    agent = agent_ctor(env.observation_space, env.action_space, cfg)
    if args.use_hdf:
        hdf, diagnostics = prepare_h5_file(args)
    gym.logger.setLevel(logging.WARN)

    COUNTER = 0
    def callback(stats):
        global COUNTER
        COUNTER += 1  
        # Print stats
        print("*********** Iteration %i ****************" % COUNTER)
        print(tabulate([ (k, v) for k, v in viewitems(stats) if np.asarray(v).size== 1 ])) #pylint: disable=W0110
        # Store to hdf5
        if args.use_hdf:
            for (stat,val) in viewitems(stats):
                if np.asarray(val).ndim==0:
                    diagnostics[stat].append(val)
                else:
                    assert val.ndim == 1
                    diagnostics[stat].extend(val)
            if args.snapshot_every and ((COUNTER % args.snapshot_every==0) or (COUNTER==args.n_iter)): 
                hdf['/agent_snapshots/%0.4i'%COUNTER] = np.array(cPickle.dumps(agent,-1))
        # Plot
        if args.plot:
            animate_rollout(env, agent, min(500, args.timestep_limit))

    run_policy_gradient_algorithm(env, agent, callback=callback, usercfg = cfg)

    # if args.use_hdf:
    #     hdf['env_id'] = env_spec.id
    #     try: hdf['env'] = np.array(cPickle.dumps(env, -1))
    #     except Exception: print("failed to pickle env") #pylint: disable=W0703
    
    env.monitor.close()
