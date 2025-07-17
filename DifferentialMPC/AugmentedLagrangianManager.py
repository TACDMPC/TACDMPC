class ConstrainedQuadCost(GeneralQuadCost):
    """
    Versione finale VMAP-compatibile della classe di costo vincolata.
    """

    def __init__(self,
                 al_manager: AugmentedLagrangianManager,
                 x_max: torch.Tensor,
                 x_min: torch.Tensor,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.al_manager = al_manager
        self.x_max = x_max.to(self.device)
        self.x_min = x_min.to(self.device)

    def objective(self, X: Tensor, U: Tensor, **kwargs) -> Tensor:
        was_unbatched = X.ndim == 2
        if was_unbatched:
            X = X.unsqueeze(0);
            U = U.unsqueeze(0)
        original_cost = super().objective(X, U, **kwargs)
        batch_size, horizon, state_dim = X.shape[0], U.shape[1], self.nx
        num_constraints_per_step = self.x_max.numel() + self.x_min.numel()
        reshaped_states = X.reshape(batch_size * (horizon + 1), state_dim)
        constraint_values = get_state_constraints(reshaped_states, self.x_max, self.x_min)
        constraint_values_horizon = constraint_values.reshape(batch_size, (horizon + 1) * num_constraints_per_step)
        al_term = self.al_manager.get_cost_term(constraint_values_horizon)
        total_cost = original_cost + al_term
        if was_unbatched:
            return total_cost.squeeze(0)
        return total_cost

    def quadraticize(self, X: Tensor, U: Tensor,
                     current_lambda: torch.Tensor,
                     current_rho: torch.Tensor,
                     **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Quadratizzazione VMAP-compatibile.
        Riceve i parametri duali specifici per questo campione del batch.
        """
        l_tau, H_tau, lN, HN = super().quadraticize(X, U, **kwargs)
        horizon, state_dim = U.shape[0], self.nx
        num_constraints = current_lambda.shape[0]

        constraint_values = get_state_constraints(X, self.x_max, self.x_min)
        activated_constraints = torch.relu(constraint_values)
        Jg = torch.cat([torch.eye(state_dim, device=self.device), -torch.eye(state_dim, device=self.device)], dim=0)

        lambda_exp = current_lambda.unsqueeze(0).expand(horizon + 1, -1)
        rho_exp = current_rho.expand(horizon + 1, num_constraints)

        common_term = (lambda_exp + rho_exp * activated_constraints)
        grad_penalty = torch.einsum("ti,ij->tj", common_term, Jg)

        rho_exp_hess = current_rho.expand(horizon + 1, -1)
        hess_penalty = torch.einsum("ti,ij,ik->tjk", rho_exp_hess, Jg, Jg)

        l_tau[:horizon, :state_dim] += grad_penalty[:-1, :]
        H_tau[:horizon, :state_dim, :state_dim] += hess_penalty[:-1, :, :]
        lN[:state_dim] += grad_penalty[-1, :]
        HN[:state_dim, :state_dim] += hess_penalty[-1, :, :]

        return l_tau, H_tau, lN, HN