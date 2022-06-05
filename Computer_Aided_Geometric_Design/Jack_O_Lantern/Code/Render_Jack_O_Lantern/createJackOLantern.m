% Author: Siddharth Palani Natarajan and Nathan Flynn

% ME 535 - Jack-O'-Lantern Surface plot using Bezier and NURBS surfaces

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Section 1 : Project Configurations
% Section 2 : Functions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% Section 1 : Project Configurations %%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; clear; clc;
delete *.stl; %Deletes all previous stl files in the current location that your matlab is pointing to.
Generate_STL = false; 
Generate_CPS = false; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%  Section 2: Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a = Generate_STL;
b = Generate_CPS;
pumpkin(a, b); %Creates the surface of the pumpkin using beizer patches 
mouth(a, b); %Creates the mouth of the pumpkin using nurbs extrusion
nose(a, b); %Creates the nose of the pumpkin using nurbs extrusion 
lefteye(a, b); %Creates the left eye of the pumpkin using nurbs extrusion 
righteye(a, b); %Creates the right eye of the pumpkin using nurbs extrusion 
Pumpkin_Stem_Surface(a, b); %Creates the stem of the pumpkin
Pumpkin_connect(a, b); %Creates the surface between the stem of the pumpkin and the pumpkin it's self.
axis off; 
view(90, 11.25);