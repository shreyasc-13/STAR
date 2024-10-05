"""
Implementation of our method.
Takes as input an (\phi, c), processes the behavior data to build teh ARP(\pi_e) and output J(\pi_e).
"""

from copy import deepcopy
import numpy as np


class STAR:
    def __init__(self, num_abstract_states, c, gamma, centroids):
        self.num_abstract_states = num_abstract_states
        self.c = c
        self.gamma = gamma
        self.centroids = centroids

    def project_data(self, centroids, states, next_states, num_states):
        """Map states to abstract states"""
        # If states are integers make them one hot
        if num_states > 1:
            states = np.stack([np.eye(num_states)[state[:, 0, 0]]
                              for state in states], axis=0)[:, :, None, :]
            next_states = np.stack([np.eye(num_states)[next_state[:, 0, 0]]
                                   for next_state in next_states], axis=0)[:, :, None, :]

        distances = np.linalg.norm(states - centroids, axis=-1)
        abstract_states = np.argmin(distances, axis=-1)

        distances = np.linalg.norm(next_states - centroids, axis=-1)
        abstract_next_states = np.argmin(distances, axis=-1)

        return abstract_states, abstract_next_states

    def estimate_MRP(self, weights, states, actions, rewards, next_states, dones):
        """
        Input: abstracted state (\phi(s), a, r) data
        Output: estimated ARP(\pi_e) [hat_p, hat_R, hat_eta, hat_term]
        """

        hat_p = np.zeros((self.num_abstract_states, self.num_abstract_states))
        hat_R = np.zeros((self.num_abstract_states,))
        hat_eta = np.zeros((self.num_abstract_states,))
        hat_term = np.zeros((self.num_abstract_states,))

        # Counts for abstract state, consecutive abstract state
        # pairs, and termination.
        num_z = np.zeros(self.num_abstract_states)
        num_z_zdash = np.zeros(
            (self.num_abstract_states, self.num_abstract_states))
        num_term = np.zeros(self.num_abstract_states)
        cum_R_z = np.zeros(self.num_abstract_states)

        first_done_mask = np.ones_like(dones) * False
        valid_traj_mask = np.ones_like(dones) * True
        for i, done_traj in enumerate(dones):
            if True in list(done_traj):
                idx = list(done_traj).index(True)
                first_done_mask[i, idx] = True
                valid_traj_mask[i, idx+1:] = False
            else:
                continue

        # Importance weighted counts computation
        for z in range(self.num_abstract_states):
            z_mask = (states == z) * valid_traj_mask

            num_z[z] = np.sum(weights[z_mask])
            cum_R_z[z] = np.sum(weights[z_mask] * rewards[z_mask])

            hat_eta[z] = z_mask[:, 0].astype(float).sum()

            done_at_z_mask = np.logical_and(z_mask, first_done_mask)
            num_term[z] = np.sum(weights[done_at_z_mask])

            for zdash in range(self.num_abstract_states):
                zdash_mask = (next_states == zdash) * valid_traj_mask

                z_zdash_mask = np.logical_and(z_mask, zdash_mask)
                num_z_zdash[z, zdash] = np.sum(weights[z_zdash_mask])

        n = states.shape[0]
        hat_eta = hat_eta/n

        # Model estimation
        # Gets all the z indices - including ones that go to zdash and terminate
        idx = np.where(num_z_zdash > 0)[0]
        for index in set(idx):
            hat_p[index] = num_z_zdash[index]/num_z[index]
            hat_R[index] = cum_R_z[index]/num_z[index]
            hat_term[index] = num_term[index]/num_z[index]

        return hat_p, hat_R, hat_eta, hat_term

    def evaluate(self, info, hat_p=None, hat_R=None, hat_eta=None, hat_term=None):
        """
        Evaluate the MRP: either in closed form or with MC rollouts.
        """
        (states,
         actions,
         rewards,
         next_states,
         dones,
         base_propensity,
         target_propensities,
         num_states) = info

        num_actions = len(target_propensities[0][0])
        actions = [np.eye(num_actions)[act] for act in actions]

        base_propensity_for_logged_action = [np.sum(np.multiply(
            bp, acts), axis=1) for bp, acts in zip(base_propensity, actions)]
        target_propensity_for_logged_action = [np.sum(np.multiply(
            tp, acts), axis=1) for tp, acts in zip(target_propensities, actions)]

        importance_weights = [np.array(p_target)/np.array(p_base) for p_target, p_base in zip(
            target_propensity_for_logged_action, base_propensity_for_logged_action)]

        c = self.c
        if not c:
            cum_importance_weights = np.array(
                [np.cumprod(rhos) for rhos in importance_weights])
        else:
            assert type(c) == int
            # truncated to c-most recent importance weights
            cum_importance_weights = np.array(
                [np.cumprod(rhos) for rhos in importance_weights])
            tcum_importance_weights = cum_importance_weights/np.roll(
                cum_importance_weights, shift=c, axis=1)
            tcum_importance_weights[:, :c] = cum_importance_weights[:, :c]
            cum_importance_weights = deepcopy(tcum_importance_weights)

        abstract_states, abstract_next_states = self.project_data(self.centroids,
                                                                  states, next_states, num_states)

        if not hat_p and not hat_R and not hat_eta and not hat_term:
            hat_p, hat_R, hat_eta, hat_term = self.estimate_MRP(
                cum_importance_weights, abstract_states, actions, rewards, abstract_next_states, dones)

        return self.evaluate_MRP(hat_p, hat_R, hat_eta, False)

    def evaluate_MRP(self, hat_p, hat_R, hat_eta, hat_term, MC_eval=False):
        if not MC_eval:
            return self._closed_form_eval(hat_p, hat_R, hat_eta, hat_term, self.gamma)
        else:
            raise "Implement MC evaluation."

    def _closed_form_eval(self, hat_p, hat_R, hat_eta, hat_term, gamma):
        """
        Compute expected return in closed form.
        Reference: Equation (6) in Appendix A.1.2.
        """
        assert len(hat_p) == self.num_abstract_states
        identity_matrix = np.identity(len(hat_p))

        termination_p_term = identity_matrix - np.diag(hat_term)
        term_p = termination_p_term @ hat_p

        if np.linalg.det(identity_matrix - gamma * term_p) == 0:
            # This condition should not evaluate to True, as the matrix
            # (identity_matrix - gamma * term_p) is invertible.
            inv_matrix = np.linalg.pinv(identity_matrix - gamma * term_p)
            print("There is likely some error in estimation.")
        else:
            inv_matrix = np.linalg.inv(identity_matrix - gamma * term_p)

        V = np.dot(inv_matrix, hat_R)

        expected_return = np.dot(V, hat_eta)

        return expected_return
