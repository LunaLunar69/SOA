%%%% Simple SOA model parameters
%%%% 
%%%% Ramon Gutierrez Castrejon, Jan 2024

clear all;
close all;

%% SOA Parameters

% Geometrical parameters
    GeomParams.L = 2.0E-3;
    GeomParams.width = 2.8E-6;
    GeomParams.depth = .25E-6;
    GeomParams.Gamma = 0.4;
    GeomParams.vg = 8.5E+7;

    GeomParams.Vol = GeomParams.width * GeomParams.depth * GeomParams.L;
    GeomParams.Aeff = GeomParams.width * GeomParams.depth / GeomParams.Gamma;

%Material Parameters
    MatParams.N0 = 0.46E24;
    MatParams.loss = 1000;
    MatParams.DiffGain = 5.3E-20;
    MatParams.A = 6.0E8;
    MatParams.B = 18.0E-16;
    MatParams.C = 1.0E-40;
    MatParams.LEF = 3.0;
    MatParams.lambda = 1550E-9;

%Signal Parameters
    SigParams.Numsymrrc = 8;       % Número de símbolos del filtro RRC
    SigParams.BitRate = 5.35e9;      % 10 Gb/s Realemente este es el symbol rate en general pero para M=2 es igual al bit rate
    SigParams.NBits = 2^14;         % This can be modified. Initially use a low number of bits for quick simulations, e.g. 2^8
    SigParams.SamplesPerBit = 32;  % Oversampling
    
   
    SigParams.TimeWindow = SigParams.NBits/SigParams.BitRate;
    SigParams.fs = SigParams.SamplesPerBit * SigParams.BitRate;   %Sample Rate
    SigParams.SamplePeriod = 1/SigParams.fs;
    
    SigParams.Groupdelay = (SigParams.Numsymrrc/2);
   % SigParams.Delay = SigParams.Numsymrrc/(2*SigParams.BitRate);   % Este es el group delay en terminos de T_M
    SigParams.Delay = int32(2*SigParams.Numsymrrc/(2*SigParams.BitRate)*SigParams.fs);

disp(GeomParams);
disp(MatParams);
disp(SigParams);

save('Params.mat','GeomParams', "MatParams", "SigParams");

