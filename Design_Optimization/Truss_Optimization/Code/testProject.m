% Author: Siddharth Palani Natarajan

% ME 548 - Design Optimization for Truss design

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

trussClass = 'truss2dMinComplianceAllConstraints';

xy = [0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100;
      0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30];

connectivity = [1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 9; 9 10; 10 11;
                11 12; 12 13; 13 14; 14 15; 15 16; 16 17; 17 18; 18 19;
                19 20; 20 21; 22 23; 23 24; 24 25; 25 26; 26 27; 27 28;
                28 29; 29 30; 30 31; 31 32; 32 33; 33 34; 34 35; 35 36;
                36 37; 37 38; 38 39; 39 40; 40 41; 41 42; 1 22; 2 23; 
                3 24; 4 25; 5 26; 6 27; 7 28; 8 29; 9 30; 10 31; 11 32; 
                12 33; 13 34; 14 35; 15 36; 16 37; 17 38; 18 39; 19 40; 
                20 41; 21 42; 23 1; 23 3; 25 3; 25 5; 27 5; 27 7; 29 7; 
                29 9; 31 9; 31 11; 33 11; 33 13; 35 13; 35 15; 37 15; 
                37 17; 39 17; 39 19; 41 19; 41 21; 43 44; 44 45; 45 46; 
                46 47; 47 48; 48 49; 49 50; 50 51; 51 52; 52 53; 53 54; 
                54 55; 55 56; 56 57; 57 58; 58 59; 59 60; 60 61; 61 62; 
                62 63; 22 43; 23 44; 24 45; 25 46; 26 47; 27 48; 28 49;
                29 50; 30 51; 31 52; 32 53; 33 54; 34 55; 35 56; 36 57; 
                37 58; 38 59; 39 60; 40 61; 41 62; 42 63; 43 23; 45 23; 
                45 25; 47 25; 47 27; 49 27; 49 29; 51 29; 51 31; 53 31; 
                53 33; 55 33; 55 35; 57 35; 57 37; 59 37; 59 39; 61 39; 
                61 41; 63 41; 64 65; 65 66; 66 67; 67 68; 68 69; 69 70; 
                70 71; 71 72; 72 73; 73 74; 74 75; 75 76; 76 77; 77 78; 
                78 79; 79 80; 80 81; 81 82; 82 83; 83 84; 43 64; 44 65; 
                45 66; 46 67; 47 68; 48 69; 49 70; 50 71; 51 72; 52 73; 
                53 74; 54 75; 55 76; 56 77; 57 78; 58 79; 59 80; 60 81; 
                61 82; 62 83; 63 84; 65 43; 65 45; 67 45; 67 47; 69 47; 
                69 49; 71 49; 71 51; 73 51; 73 53; 75 53; 75 55; 77 55; 
                77 57; 79 57; 79 59; 81 59; 81 61; 83 61; 83 63; 85 86; 
                86 87; 87 88; 88 89; 89 90; 90 91; 91 92; 92 93; 93 94; 
                94 95; 95 96; 96 97; 97 98; 98 99; 99 100; 100 101; 101 102; 
                102 103; 103 104; 104 105; 64 85; 65 86; 66 87; 67 88; 68 89; 
                69 90; 70 91; 71 92; 72 93; 73 94; 74 95; 75 96; 76 97; 77 98; 
                78 99; 79 100; 80 101; 81 102; 82 103; 83 104; 84 105; 85 65; 
                87 65; 87 67; 89 67; 89 69; 91 69; 91 71; 93 71; 93 73; 95 73; 
                95 75; 97 75; 97 77; 99 77; 99 79; 101 79; 101 81; 103 81; 
                103 83; 105 83; 106 107; 107 108; 108 109; 109 110; 110 111;
                111 112; 112 113; 113 114; 114 115; 115 116; 116 117; 117 118;
                118 119; 119 120; 120 121; 121 122; 122 123; 123 124; 124 125;
                125 126; 85 106; 86 107; 87 108; 88 109; 89 110; 90 111;
                91 112; 92 113; 93 114; 94 115; 95 116; 96 117; 97 118; 98 119;
                99 120; 100 121; 101 122; 102 123; 103 124; 104 125; 105 126;
                85 107; 87 107; 87 109; 89 109; 89 111; 91 111; 91 113; 93 113;
                93 115; 95 115; 95 117; 97 117; 97 119; 99 119; 99 121; 101 121;
                101 123; 103 123; 103 125; 105 125; 127 128; 128 129; 129 130; 
                130 131; 131 132; 132 133; 133 134; 134 135; 135 136; 136 137; 
                137 138; 138 139; 139 140; 140 141; 141 142; 142 143; 143 144; 
                144 145; 145 146; 146 147; 106 127; 107 128; 108 129; 109 130; 
                110 131; 111 132; 112 133; 113 134; 114 135; 115 136; 116 137; 
                117 138; 118 139; 119 140; 120 141; 121 142; 122 143; 123 144; 
                124 145; 125 146; 126 147; 127 107; 129 107; 129 109; 131 109; 
                131 111; 133 111; 133 113; 135 113; 135 115; 137 115; 137 117; 
                139 117; 139 119; 141 119; 141 121; 143 121; 143 123; 145 123; 
                145 125; 147 125]';

t = feval(trussClass,xy,connectivity); 
A = 3060/(sum(t.myL)*7700); % Area Constraint
t = t.assignE(2e11); % Young's Modulus
t = t.assignA(A); % Area
t = t.fixXofNodes(1); % Nodes fixed in x
t = t.fixYofNodes([1 21]); % Nodes fixed in y
t = t.applyForce([2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20],[0;-100000/19]); % Nodes where force is applied
t.plot(1);
tic
t = t.optimize();
toc
optimizedArea = t.myArea;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Results
figure; t.plot(); hold on; t.plotDeformed();
disp('Initial Compliance: '); disp(t.myInitialCompliance)
disp('Final Compliance: '); disp(t.myFinalCompliance);
disp('Minimum bar length');disp(min(t.myL))
disp('Maximum bar length');disp(max(t.myL))
disp('Total mass before Optimization');disp(sum(t.myInitialMass))
disp('Total mass after Optimization');disp(sum(t.myFinalMass))
disp('Maximum Deformation before Optimization');disp(t.myInitialDeformation)
disp('Maximum Deformation after Optimization');disp(t.myFinalDeformation)
disp('Maximum Stress before Optimization');disp(max(t.myInitialStress))
disp('Maximum Stress after Optimization');disp(max(t.myFinalStress))