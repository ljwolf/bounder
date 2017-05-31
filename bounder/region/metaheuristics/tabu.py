from .abstracts import MetaHeuristic
from ..abstracts import move
import numpy as np
from tqdm import tqdm
import itertools as it
try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO

def _filelogger(outfile):
    def logger(string):
        with open(outfile, 'w') as out:
            out.write(str(string))
            out.write('\n')
    return logger
def _selflogger(output):
    def logger(string):
        output = '\n'.join((output, str(string)))
    return logger

class Tabu(MetaHeuristic):
    def __init__(self, remember=True, max_size=None):
        MetaHeuristic.__init__(self)
        self.remember = remember
        self.tabu_list = []
        self._trace = []
        self.max_size = max_size
        def noop(*args, **kw):
            pass
        self._log = noop
        #if log is '':
        #    self.log = ''
        #    self._log = _selflogger(self.log)
        #elif log.lower() == 'print':
        #    self._log = print
        #else:
        #    self._log = _filelogger(log)

    def refine(self, passes = 1, rounds = 0,
               target_label=None, swap=False, 
               progressbar = False):
        """
        Conduct a pass of a Tabu Search on a given regionalization problem. 

        Passes : number of times to iterate through the tabu replacement
        rounds : number of times to do a single pass on each region
        ignore_objective: whether to just swap on feasibility or not. 
        target_label : the label to focus the refinement on
        swap : whether to swap observations between regions or just move them.
        """
        if not progressbar:
            def noop(seq):
                return seq
            tqdm_ = noop
        else:
            tqdm_ = tqdm
        if self.max_size is None:
            self.max_size = self.boundary_alist.shape[0] // self.n_clusters
        on_pass = 0
        print('starting passes')
        if rounds > 0:
            repeater = np.arange(self.n_clusters).tolist()*rounds
            for target in tqdm_(repeater):
                self.refine(passes = 1, rounds=0, 
                            target_label = target, swap=swap,
                            progressbar = False)
            return
        for _ in tqdm_(range(passes)):
            if target_label is not None: 
                label = target_label
            else:
                label = np.random.randint(0,self.n_clusters)
            print('refining {}'.format(label))
            candidates = list(self._generate_candidates(label))
            print(candidates)
            best_move = move(None, None, None, None,
                             -np.inf, -np.inf, 
                             np.inf, np.inf,
                             False, 'initial')
            for focal, neighbor in candidates:
                if (focal, neighbor) not in self.tabu_list:
                    orig_label = self._data.ix[focal].current_labels
                    candidate_label = self._data.ix[neighbor].current_labels
                    print('considering {} ({}) -> {} ({})'\
                              .format(focal, orig_label, neighbor, candidate_label))
                    local = self._data.copy()
                    lgrouper = local.groupby('current_labels')
                    current_objective = self.score(local)
                    current_connected= self.connected(local)
                    current_feasible = self.feasible(local)
                    current_slack = self.slack(local)
                    current_valid = current_connected and current_feasible
                    local.ix[focal, 'current_labels'] = candidate_label
                    new_objective = self.score(local)
                    new_connected = self.connected(local)
                    new_feasible = self.feasible(local)
                    new_slack = self.slack(local)
                    new_valid = new_connected and new_feasible
                    current_burden = current_slack * (current_slack > 0).astype(int)
                    current_burden = np.sum(current_burden.sum(axis=1))
                    new_burden = new_slack * (new_slack > 0).astype(int)
                    new_burden = np.sum(new_burden.sum(axis=1))
                    print((new_objective, current_objective, new_burden, current_burden))
                    if new_valid and (not current_valid):
                        print('valid move')
                        best_move = move(focal, orig_label, 
                                             neighbor, candidate_label, 
                                             new_objective, current_objective,
                                             new_burden, current_burden,
                                             new_valid, 'validity')
                    elif new_connected and (new_burden < current_burden):
                        print('slack reduction {}'.format(
                              np.hstack((new_burden, current_burden))))
                        #evaluate slacks and accept if it improves the slacks
                        print(new_burden, best_move.new_burden)
                        if new_burden < best_move.new_burden:
                            best_move = move(focal, orig_label, 
                                             neighbor, candidate_label, 
                                             new_objective, current_objective,
                                             new_burden, current_burden,
                                             new_valid, 'slack')
                    elif new_valid and (current_objective < new_objective):
                        print('obj reduction {}'.format(
                              new_objective - current_objective))
                        if best_move.new_objective < new_objective:
                            best_move = move(focal, orig_label, 
                                             neighbor, candidate_label,
                                             new_objective, current_objective,
                                             new_burden, current_burden,
                                             new_valid, 'objective')
            if best_move.type is not "initial":
                self._data.ix[best_move[0], 'current_labels'] = best_move.neighbor_label
                if self.remember:
                    self._trace.append(best_move)
                else:
                    self._trace = [best_move]
                self.tabu_list.append((best_move.focal_id, 
                                       best_move.neighbor_id))
                if len(self.tabu_list) > self.max_size:
                    self.tabu_list.pop(0)

    def _generate_candidates(self, label):
        """
        return the entries of an adjacency list that 
        are on the boundary of the region in play. 
        """
        mask = self.boundary_alist.focal.isin(self._data.query(\
                                              'current_labels == {}'.format(label)).index.tolist())
        #mask &= ~self.boundary_alist.neighbor.isin(self._data.query(\
        #                                      'current_labels == {}'.format(label)).index.tolist())
        return self.boundary_alist[mask][['focal', 'neighbor']].values
