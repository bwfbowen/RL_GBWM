classdef CustomGAgent < rl.agent.CustomAgent
    % CustomGAgent: Creates an G-Learning Agent for discrete state action.
    
    
    %% Public Properties
    properties
        % G[state, action]
        G = [0, 0]

        % N[state, action]
        N = [0, 0]

        % actions
        actions

        % rho the prior policy
        rho

        % k: param for adjusting beta.
        k

        % total T steps
        T

        % epsilon for exploration
        epsilon

        % discount
        discount

        
    end
    
    
    %% MAIN METHODS
    methods
        % Constructor
        function obj = CustomGAgent(num_state, num_action, k, epsilon, obs_info, act_info, T, discount)

            % Call the abstract class constructor
            obj = obj@rl.agent.CustomAgent();

            % Set the G and N matrices
            obj.G = zeros(num_state, num_action);
            obj.N = zeros(num_state, num_action);
            obj.rho = ones(num_state, num_action) ./ num_action;
            obj.actions = 1:num_action;

            obj.k = k;
            obj.epsilon = epsilon;
            obj.T = T;
            obj.discount = discount;

            % Define the observation and action spaces
            obj.ObservationInfo_ = obs_info;
            obj.ActionInfo_ = act_info;

        end

        function flat_idx = two2one(obj, Observation)
            % Observation is a cell of array (1 times 2), convert to 1d
            % array
            obs_arr = Observation{:};
            flat_idx = (obs_arr - [1, 0]) * [obj.T; 1];
        end 

        function alpha = schedule_alpha(obj, Observation, action)
            flat_idx = two2one(obj, Observation);
            alpha = obj.N(flat_idx, action) ^ -0.8;
        end

        function beta = schedule_beta(obj, Observation)
            obs_arr = Observation{:};
            beta = obj.k * obs_arr(2);
        end

    end
    
    %% Implementation of abstract parent protected methods
    methods (Access = protected)
        function action = getActionWithExplorationImpl(obj,Observation)
            % Given the current observation, select an action with
            % exploration. Here uses epsilon-greedy strategy
            action = getActionImpl(obj,Observation, obj.epsilon);
        end
        % learn from current experiences, return action with exploration
        % transition = {state,action,reward,nextstate,isdone}
        function action = learnImpl(obj, transition)
            % update with exp
            Observation = transition{1};
            state = two2one(obj, Observation);
            action_ = transition{2}{:};
            cost = -transition{3};
            nxt_state = two2one(obj, transition{4});
            obj.N(state, action_) = obj.N(state, action_) + 1;
            alpha = schedule_alpha(obj, Observation, action_);
            beta = schedule_beta(obj, Observation);
            temp = sum(obj.rho(nxt_state, :) .* exp(-beta .* obj.G(nxt_state, :)));
            td_target = cost - (obj.discount / beta) * log(temp);
            td_delta = td_target - obj.G(state, action_);
            obj.G(state, action_) = obj.G(state, action_) + alpha * td_delta;
            % Find and return an action with exploration
            action = getActionWithExplorationImpl(obj,transition{4});
            
        end
           
        
        % Action methods
        function action = getActionImpl(obj,Observation, epsilon)
            % By default, apply greedy strategy
            if nargin < 3
                epsilon = 0;
            end
            % Given the current state of the system, return an action.
            num_action = size(obj.G, 2);
            action_probs = ones(num_action, 1) .* epsilon ./ num_action;
            tmp_idx = two2one(obj, Observation);
            [~, best_action] = max(obj.G(tmp_idx, :));
            action_probs(best_action) = action_probs(best_action) + (1.0 - epsilon);
            action = randsample(obj.actions, 1, true, action_probs);
        end
 
    end
        
end
