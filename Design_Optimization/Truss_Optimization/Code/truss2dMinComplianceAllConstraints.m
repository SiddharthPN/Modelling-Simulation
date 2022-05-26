% Author: Siddharth Palani Natarajan

% ME 548 - Design Optimization for Truss design

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef truss2dMinComplianceAllConstraints < truss2d 

    properties(GetAccess = 'public', SetAccess = 'private')
        myInitialVolume;
        myInitialArea;
        myInitialCompliance;
        myInitialMass;
        myInitialDeformation;
        myInitialStress;
        myFinalArea;
        myFinalCompliance;
        myFinalVolume;
        myFinalMass;
        myFinalStress;
        myFinalDeformation;
        myYieldStress;
        myLambda;
    end
    methods
        function obj = truss2dMinComplianceAllConstraints(xy,connectivity)
            obj = obj@truss2d(xy,connectivity);
            obj.myYieldStress(1:obj.myNumTrussBars) = 100e6;
        end
        function obj = assignYieldStress(obj,yieldStress,members)

            if (nargin == 2)
                members = 1:obj.myNumTrussBars;
            else
                assert(max(members) <= obj.myNumTrussBars);
                assert(min(members) >=  1);
            end
            obj.myYieldStress(members) = yieldStress;
        end 
        function JRelative = complianceObjective(obj,x)
            Area = x.*obj.myInitialArea;
            obj = obj.assignA(Area);
            obj = obj.assemble();
            obj = obj.solve();
            J = obj.getCompliance();
            JRelative = J/obj.myInitialCompliance;
        end  

        function [cineq,ceq] = stressConstraint(obj,x)
            Area = x.*obj.myInitialArea;
            obj = obj.assignA(Area);
            obj = obj.assemble();
            obj = obj.solve();
            nConstraints = 2*obj.myNumTrussBars;
            cineq = zeros(1,nConstraints);
            constraint = 1;
            for m = 1:obj.myNumTrussBars
                cineq(constraint) = obj.myStress(m)/obj.myYieldStress(m)-1; % Tensile Stress
                cineq(constraint+1) = -obj.myStress(m)/obj.myYieldStress(m) -1; % Compressive Stress
                constraint = constraint+2; 
            end 
            ceq = [];        
        end
        function obj = initialize(obj)
            obj.myInitialArea = obj.myArea;
            obj.myInitialVolume = sum(obj.myArea.*obj.myL);
            obj = obj.assemble();
            obj = obj.solve();
            obj.myInitialCompliance =  obj.getCompliance();
            obj.myInitialMass = sum((obj.myInitialArea.*obj.myL))*7700;
            obj.myInitialDeformation = obj.myDeformationMax;
            obj.myInitialStress = obj.myStress;
        end
        function processLambda(obj)
            ineqnonlin = obj.myLambda.ineqnonlin;
            maxValue = max(abs(ineqnonlin));
            ineqnonlin = ineqnonlin/maxValue;
            for m = 1:obj.myNumTrussBars
                if (ineqnonlin(2*m-1) > 0.0001)
                    disp(['Bar ' num2str(m) ': Tensile stress active']);
                elseif (ineqnonlin(2*m) > 0.0001)
                    disp(['Bar ' num2str(m) ': Compressive stress active']);
                else
                    disp(['Bar ' num2str(m) ': Stress inactive']);
                end
            end
        end
        function obj = optimize(obj)
            options = optimset('fmincon');
            options.MaxFunEvals = 10000000;
            options.Iterations = 1000000;
            obj = obj.initialize();
            x0 = ones(1,obj.myNumTrussBars); 
            LB = 1e-12*ones(1,obj.myNumTrussBars);
            AinEq = (obj.myInitialArea.*obj.myL)/obj.myInitialVolume;
            BinEq = 1;
            [xMin,~,~,~,Lambda]  = fmincon(@obj.complianceObjective,x0, ...
                   AinEq,BinEq,[],[],LB,[],@obj.stressConstraint,options);
            obj = obj.assignA(xMin.*obj.myInitialArea);
            obj = obj.assemble();
            obj = obj.solve();
            obj.myFinalArea= obj.myArea;
            obj.myFinalVolume = sum(obj.myArea.*obj.myL);
            obj.myFinalCompliance =  obj.getCompliance(); 
            obj.myFinalMass = sum(obj.myArea.*obj.myL)*7700;
            obj.myFinalDeformation = obj.myDeformationMax;
            obj.myFinalStress = obj.myStress;
            obj.myLambda = Lambda;
        end
    end
end

