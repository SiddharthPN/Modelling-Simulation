       dip1 = 4; dip2 =1; h1 = 5.5; h2 = 4.125; h3 = 2.75; h4 =1.375; h5 = 0; d = 0.885; r = 3.343; rn1 = 1*pi/96; r0 = 2*pi/96; r1 = 3*pi/96; r2 = 9*pi/96; r3 = 15*pi/96; r4 = 21*pi/96; r5 = 22*pi/96; r6 = 23*pi/96; r7 = 24*pi/96; i = 1/2; verts = [ 0 0 dip1; % Start
            0 0 dip1;
            0 0 dip1;
            0 0 dip1; % End
            d*cos(r1) d*sin(r1) h1; % Start
            cos(r2) sin(r2) h1;
            cos(r3) sin(r3) h1;
            d*cos(r4) d*sin(r4) h1; % End
            d*r*cos(r1) d*r*sin(r1) h2; % Start
            r*cos(r2) r*sin(r2) h2;
            r*cos(r3) r*sin(r3) h2;
            d*r*cos(r4) d*r*sin(r4) h2; % End
            d*r*cos(r1) d*r*sin(r1) h3; % Start
            r*cos(r2) r*sin(r2) h3;
            r*cos(r3) r*sin(r3) h3;
            d*r*cos(r4) d*r*sin(r4) h3; % End
            d*r*cos(r1) d*r*sin(r1) h4; % Start
            r*cos(r2) r*sin(r2) h4;
            r*cos(r3) r*sin(r3) h4;
            d*r*cos(r4) d*r*sin(r4) h4; % End
            d*cos(r1) d*sin(r1) h5; % Start
            cos(r2) sin(r2) h5;
            cos(r3) sin(r3) h5;
            d*cos(r4) d*sin(r4) h5; % 
            0 0 dip2;
            0 0 dip2;
            0 0 dip2;
            0 0 dip2; %
            d*cos(r0) d*sin(r0) h1; %%%%%%%%%%%%%% 
            cos(rn1) sin(rn1) h1;
            cos(0) sin(0) h1;  % 1
            d*r*cos(r0) d*r*sin(r0) h2;
            r*cos(rn1) r*sin(rn1) h2;
            r*cos(0) r*sin(0) h2; % 2
            d*r*cos(r0) d*r*sin(r0) h3;
            r*cos(rn1) r*sin(rn1) h3;
            r*cos(0) r*sin(0) h3; % 3 
            d*r*cos(r0) d*r*sin(r0) h4;
            r*cos(rn1) r*sin(rn1) h4;
            r*cos(0) r*sin(0) h4; % 4
            d*cos(r0) d*sin(r0) h5;
            cos(rn1) sin(rn1) h5;
            cos(0) sin(0) h5; %%%%%%%%%%
            d*cos(r5) d*sin(r5) h1;
            cos(r6) sin(r6) h1;
            cos(r7) sin(r7) h1;  % 1
            d*r*cos(r5) d*r*sin(r5) h2;
            r*cos(r6) r*sin(r6) h2;
            r*cos(r7) r*sin(r7) h2; % 2
            d*r*cos(r5) d*r*sin(r5) h3;
            r*cos(r6) r*sin(r6) h3;
            r*cos(r7) r*sin(r7) h3; % 3 
            d*r*cos(r5) d*r*sin(r5) h4;
            r*cos(r6) r*sin(r6) h4;
            r*cos(r7) r*sin(r7) h4; % 4
            d*cos(r5) d*sin(r5) h5;
            cos(r6) sin(r6) h5;
            cos(r7) sin(r7) h5;
            ]; 