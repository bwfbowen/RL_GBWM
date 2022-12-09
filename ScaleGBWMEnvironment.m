classdef ScaleGBWMEnvironment < rl.env.MATLABEnvironment
    %GBWMENVIRONMENT: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties    
        % pwgt
        pwgt = [0, 0]
        pret = [0]
        prsk = [0]

        % simulate params
        simulate_n_periods = 1
        simulate_dt = 1
        simulate_n_trials = 1

        % Efficient frontier
        ef = [0, 0]
        
        % Total time T
        T = 0
        
        % Goal G
        G = 0

        % cash
        cash = []
        
        % Grid of wealth points, from small to large
        grid = [0; 0]
        grid_idx = [1, 2]
        
        
        % Current time step t
        t = 1
        
        % Current wealth index in grid
        cur_w = 0
        w0_idx = 0

        % Sample time
        Ts = 0.02
        
        % Reward each time step the wealth reached the goal
        reward_fulfill = 1
        
        % Reward each time step for not reaching the goal
        reward_unfulfill = 0
        
        % Gamma for reward discount
        gamma = 0.95
    end
    
    properties
        % Initialize system state [log(w), t, log(goal)]', with MinMax Scaler
        State = zeros(3,1)
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = ScaleGBWMEnvironment(G, T, grid, cash, w0_idx, pwgt, pret, prsk, gamma, simulate_n_periods, simulate_dt, simulate_n_trials)
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([2 1]);

            ObservationInfo.Name = 'GBWM States';
            ObservationInfo.Description = 'wealth, time';
            
            % Initialize Action settings   
            ActionInfo = rlFiniteSetSpec(1:length(pwgt));
            ActionInfo.Name = 'mu-sig pair Action';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            % Initialize env variables
            this.w0_idx = w0_idx;
            this.G = G;
            this.T = T;
            this.grid = grid;
            this.grid_idx = 1:length(grid);
            this.cash = cash;
            this.pwgt = pwgt;
            this.pret = pret;
            this.prsk = prsk;
            this.simulate_n_periods = simulate_n_periods;
            this.simulate_dt = simulate_dt;
            this.simulate_n_trials = simulate_n_trials;
            this.cur_w = w0_idx;
            this.gamma = gamma;
            
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this, Action)
            LoggedSignals = [];
            
            GBM = gbm(this.pret(Action), this.prsk(Action), ...
                "StartState", this.grid(this.cur_w), "StartTime", this.t);
            [simulate_samples, ~] = simulate(GBM, ...
                this.simulate_n_periods, "DeltaTime", this.simulate_dt, 'nTrials', this.simulate_n_trials);
            simulate_samples = squeeze(simulate_samples(end, 1, :));
            nxt_wealth_sample = randsample(simulate_samples, 1, true);
            this.t = this.t + 1;
            nxt_wealth_sample = nxt_wealth_sample + this.cash(this.t);
            % nxt_wealth_sample + cash(this.t);
            difference = this.grid - nxt_wealth_sample;
            [~, nxt_w] = min(abs(difference));

            this.cur_w = nxt_w;
            Observation = construct_state(this, this.t, this.cur_w);

            % Update system states
            this.State = Observation;
            
            % Check terminal condition
            IsDone = construct_done(this);
            this.IsDone = IsDone;
            
            % Get reward
            Reward = construct_reward(this);
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            this.t = 1;
            this.cur_w = this.w0_idx;
            this.IsDone = construct_done(this);
            
            InitialObservation = construct_state(this, this.t, this.cur_w);
            this.State = InitialObservation;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods               
        % Helper methods to create the environment
        function Observation = construct_state(this, t, w_idx)
            
            t_norm = t ./ this.T;
            w_norm = (log(this.grid(w_idx)) - log(this.grid(1))) ./ (log(this.grid(end)) - log(this.grid(1)));
            Observation = [w_norm, t_norm].';

        end

        % Reward function
        function Reward = construct_reward(this)
            if this.grid(this.cur_w) >= this.G
                Reward = this.reward_fulfill;
            else
                Reward = this.reward_unfulfill;
            end   
            Reward = Reward .* this.gamma .^ this.t;
        end

        function IsDone = construct_done(this)
            IsDone = this.t > this.T - 1;
        end
        
        % (optional) Visualization method
        function plot(this)
            % Initiate the visualization
            
            % Update the visualization
            envUpdatedCallback(this)
        end
        
        
    end
    
    methods (Access = protected)
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
        end
    end
end
