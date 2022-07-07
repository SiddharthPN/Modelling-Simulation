classdef beamRBDO
    properties(GetAccess = 'public', SetAccess = 'private')
        myBetaTarget
    end
    methods
        function obj = beamRBDO(betaTarget)
            if (nargin == 0)
                betaTarget = 3;
            end
            obj.myBetaTarget = betaTarget;
        end
        %% outer obj func
        function [f df]= outerObjective(~,xVec)
            f = xVec(1)*xVec(2);
            if nargout>1
                df=[xVec(2);xVec(1)];
            end
        end
        %% outer constraint func
        function [cineq,ceq,DCineq,DCeq] = outerConstraint(obj,xVec)
            %disp(xVec)
            meanx=[1000 500 40000]; %mean values of Y load, Z load, and yield strength
            stdx=sqrt([100 100 2000]); %variance transformed to the std deviation
            beta= obj.innerOptimization(xVec);
            u=beta.uVec;
            beta_t= obj.myBetaTarget;
            cineq= (beta.beta-beta_t);
            ceq=[];
            DCeq=[];
            if nargout >2
                U1=u(1)*stdx(1)+meanx(1);
                U2=u(2)*stdx(2)+meanx(2);
                U3=u(3)*stdx(3)+meanx(3);
                delta_G_Xvec1=- (600*U1)/(xVec(1)^2*xVec(2)^2) - (1200*U2)/(xVec(1)^3*xVec(2));
                delta_G_Xvec2=- (1200*U1)/(xVec(1)*xVec(2)^3) - (600*U2)/(xVec(1)^2*xVec(2)^2);
                DCineq=[delta_G_Xvec1;delta_G_Xvec2];
            end
        end
        %% inner loop
        function beta = innerOptimization(obj,xVec)
            %disp(xVec)
            u0=[0.0001 0.0001 0.0001]; % U-space solution, standard normal variables
            options=optimoptions('fmincon',...
                'SpecifyObjectiveGradient',false,'SpecifyConstraintGradient',false,'Display','off');
            [beta.uVec,beta.beta]=fmincon(@(uVec) obj.innerObjective(uVec,xVec),u0,[],[],[],[],[],[],...
                @(uVec)obj.innerConstraint(uVec),options);
        end
        %% inner obj func
        function [f Df] = innerObjective(~,u,xVec)
            meanx=[1000 500 40000]; %mean values of Y load, Z load, and yield strength
            stdx=sqrt([10 10 2000]); %variance transformed to the std deviation
            % U space transformation
            U1=u(1)*stdx(1)+meanx(1);
            U2=u(2)*stdx(2)+meanx(2);
            U3=u(3)*stdx(3)+meanx(3);
            % U-space (perf func - Sy) Solution
            f=600*U1/(xVec(1)*xVec(2)^2)+600*U2/(xVec(1)^2*xVec(2))-U3;
            if nargout>1
                Df= [600*stdx(1)/(xVec(1)*xVec(2)^2)
                    600*stdx(2)/(xVec(1)*xVec(2)^2)
                    -stdx(3)];
            end
        end
        %% inner constraint (beta)
        function [cineq,ceq,Dc,Dceq] = innerConstraint(obj,u)
            ceq=sqrt(u(1)^2+u(2)^2+u(3)^2)-obj.myBetaTarget;
            cineq=[];
            Dc=[];
            if nargout  > 2
                Dceq = [ u(1)/(u(1)^2 + u(2)^2 + u(3)^2)^(1/2),
                    u(2)/(u(1)^2 + u(2)^2 + u(3)^2)^(1/2),
                    u(3)/(u(1)^2 + u(2)^2 + u(3)^2)^(1/2)];
            end
        end
        %% main loop
        function obj = optimize(obj)
            lb = [0.00001,0.00001];
            ub = [];
            A = [];
            b = [];
            Aeq = [];
            beq = [];
            x0 = [2.5,2.5]; % initial guess for design variables
            options = optimoptions('fmincon','Display','iter','Algorithm','sqp', ...
                'SpecifyObjectiveGradient',false,'SpecifyConstraintGradient',false,'Display', 'iter');
            [x fval exitflag output] = fmincon(@(xVec)obj.outerObjective(xVec),x0,A,b,Aeq,beq,lb,ub,...
                @(xVec)obj.outerConstraint(xVec),options);
            output
            fprintf('Optimum design variables values, w=%g\t t=%g\n',x(1),x(2))
        end
    end
end
