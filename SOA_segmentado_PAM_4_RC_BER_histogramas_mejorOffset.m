% Programa que modela un SOA (segmentado) en una configuración back-to-back 
% cuando la corriente de entrada al SOA esta modulada con formato PAM-4
% Intento con código Gray

clear; clc;
Initialization;

tTotal = tic;     % INICIO: tiempo total

rng(4);                                    % Seed. 
Bits = randi([0 1],SigParams.NBits,1);     % Bits PRBS

%%  PAM-4: Gray mapping and electric waveform
tPAM = tic;

UI   = SigParams.SamplesPerBit;    % samples per bit (PAM-2) / per symbol (PAM-4)

% === 1) bits -> símbolos PAM-4 (Gray) ===
% Gray: 00->0, 01->1, 11->2, 10->3
NSyms = floor(length(Bits)/2);
Bits  = Bits(1:2*NSyms);               
b2 = Bits(1:2:2*NSyms-1);
b1 = Bits(2:2:2*NSyms);

sym = zeros(NSyms,1);
mask = (b2==0 & b1==0); sym(mask)=0;
mask = (b2==0 & b1==1); sym(mask)=1;
mask = (b2==1 & b1==1); sym(mask)=2;
mask = (b2==1 & b1==0); sym(mask)=3;

sym2 = filter([1 1],1,upsample(2*sym-3,2)); % Nueva Input del SOA (INPUT)

SigParams.NSamples = NSyms * UI;


% === 2) Forma de onda "eléctrica" continua a Fs ===

% Tren de impulsos (un impulso por símbolo) a Fs
%ElecImp = zeros(SigParams.NSamples,1);
%ElecImp(1:UI:end) = double(sym);

ElecImp = zeros(SigParams.NSamples,1);
ElecImp(floor(UI/2)+1 : UI : end) = double(sym);  % ahora símbolo k ocurre en off=UI/2


% Filtro Raised-Cosine (RC)
beta_rc = 0.3;       % roll-off (0..1)
span_rc = 8;         % span en símbolos (par -> group delay entero)

h_rc = rcosdesign(beta_rc, span_rc, UI, 'normal');   % RC
h_rc = h_rc(:);

FilteredSignal = conv(ElecImp, h_rc, 'same');               
FilteredSignal = FilteredSignal(1:SigParams.NSamples,1);

TimeVec = ((0:SigParams.NSamples-1)*SigParams.SamplePeriod)';
TimeVec2 = ((0:length(sym2)-1)*0.5/SigParams.BitRate)'; % Nuevo Vector de Tiempo (TIEMPO)

% Ajuste de I_bias e I_swing para que I esté en el rango deseado
Imin = 0.25; Imax = 0.65;
gd_rc = span_rc*UI/2;
x_ss  = FilteredSignal(gd_rc+1:end-gd_rc);          

x_lo = prctile(x_ss,0.1);
x_hi = prctile(x_ss,99.9);

m = (Imax - Imin)/(x_hi-x_lo);
c = Imin - m*x_lo;

I = c + m.*FilteredSignal;   
I = I(:);

tPAM_s = toc(tPAM);
fprintf('\nTiempo de ejecucion sección PAM-4: %.3f s (%.3f min)\n', tPAM_s, tPAM_s/60);

%% Optical input. CW
EField_in = CWLaser(SigParams.NSamples, SigParams.SamplePeriod);
PP = abs(EField_in).^2;       % Potencia del láser CW en [W]

%% Solución del modelo segmentado del SOA

num_segments    = 10;                          % Número de segmentos
segment_length  = GeomParams.L / num_segments; % Longitud de cada segmento

P = zeros(length(TimeVec), num_segments + 1);  % P(t) por segmento
P(:, 1) = PP;                                  % Potencia de entrada al 1er segmento

% Densidad de portadores por segmento
N = zeros(length(TimeVec), num_segments);

EField = EField_in;  % campo que se propaga segmento por segmento

for k = 1:num_segments

    % Condición inicial para la densidad de portadores
    y0 = 0.4e24; % aproximación inicial

    % Resolver ODE temporal en el segmento k con RK4 explícito en malla uniforme
    N(:, k) = SOA_N_RK4_uniform(TimeVec, I, P(:, k), y0, GeomParams, MatParams, segment_length);

    % Ganancia efectiva del segmento para todos los tiempos
    Gan = (GeomParams.Gamma * MatParams.DiffGain * (N(:, k) - MatParams.N0));

    % Propagación del campo: ganancia + atenuación (por muestra de tiempo)
    for t_idx = 1:length(TimeVec)
        argumento_ganancia   = 0.5 * (1 - 1i * MatParams.LEF) * segment_length * Gan(t_idx);
        argumento_atenuacion = -0.5 * MatParams.loss * segment_length;

        EField(t_idx) = EField(t_idx) .* exp(argumento_ganancia) .* exp(argumento_atenuacion);
    end

    % Debug opcional
    fprintf('Segmento %d: mean(Gan)=%.3e, EField_end=%.3e%+.3ei\n', ...
        k, mean(Gan), real(EField(end)), imag(EField(end)));

    % Potencia a la salida del segmento k
    P(:, k+1) = abs(EField).^2;
end

%% Output power (aún sin alinear ni quitar transitorio)
P_out   = P(:, end);
phi_out = unwrap(angle(EField)); 

%% Quitar transitorio y retardo del SOA (alinear P_out con I) 

Nskip_sym = 2;  % símbolos a saltar (para evitar el transitorio al estimar delay)
Nskip_sym = 0;  % MODIFICADO por PTF


% === 1) Segmento estable para estimar retardo ===
start0 = 1 + Nskip_sym*UI;
Nsym0  = floor((SigParams.NSamples - start0 + 1)/UI);
end0   = start0 + Nsym0*UI - 1;

if Nsym0 < 5
    error('Muy pocos símbolos para estimar delay. Reduce Nskip_sym o aumenta NBits.');
end

seg0 = start0:end0;

x_ref0 = I(seg0);      x_ref0 = x_ref0(:);
x_out0 = P_out(seg0);  x_out0 = x_out0(:);

% === 2) Estimar retardo entero (muestras) ===
d_soa = finddelay(x_ref0 - mean(x_ref0), x_out0 - mean(x_out0));

fprintf('\nDelay estimado del SOA: d_soa = %d muestras (%.3e s)\n', ...
    d_soa, d_soa*SigParams.SamplePeriod);

% === 3) Alinear ===
Ns = numel(TimeVec);
if d_soa >= 0
    idxI = 1:(Ns - d_soa);
    idxP = (1 + d_soa):Ns;
else
    d = -d_soa;
    idxI = (1 + d):Ns;
    idxP = 1:(Ns - d);
end

I_al     = I(idxI);
P_out_al = P_out(idxP);
t_al     = TimeVec(idxI);

% === 4) Quitar transitorio ya en señales alineadas ===
Ns_al = numel(t_al);
start_trim = 1 + Nskip_sym*UI;

if start_trim >= Ns_al
    error('Transitorio demasiado grande después de alinear. start_trim=%d, Ns_al=%d', start_trim, Ns_al);
end

idx_trim   = start_trim:Ns_al;
t_trim     = t_al(idx_trim);
I_trim     = I_al(idx_trim);
P_out_trim = P_out_al(idx_trim);
P_out_trim2 = P_out_trim(1:16:end); % Nueva salida del SOA (OUTPUT_new)


%% Graficas de I y P_out alineadas y sin transitorio

figure
subplot(2,1,1)
plot(t_trim, I_trim)
grid on
ylabel('Driving Current [A]'); xlabel('Time [s]')
title('I(t) alineada (sin transitorio)')

subplot(2,1,2)
plot(t_trim, 1e3*P_out_trim)
grid on
ylabel('P_{out} [mW]'); xlabel('Time [s]')
title('P_{out}(t) alineada con I (sin delay SOA) y sin transitorio')

%% Potencia promedio (SOBRE señal alineada y sin transitorio)
P_out_trim_mW = 1e3*P_out_trim;
Pavg_mW  = mean(P_out_trim_mW);
Pavg_dBm = 10*log10(Pavg_mW);

fprintf('\nPavg_mW  (trim) = %.6g mW\n',  Pavg_mW);
fprintf('Pavg_dBm (trim) = %.6f dBm\n', Pavg_dBm);

%% SER y BER + Histogramas de P_s (mejor instante de decisión por barrido de offset)

% Cada muestra de P_out_trim corresponde a un índice original:
orig_idx_surv = idxI(idx_trim);

% --- Barrido de "fase" (offset de muestreo dentro del símbolo) ---
% off = 0..UI-1  => muestrear en (k-1)*UI + off + 1  (índice ORIGINAL)
off_vec = 0:UI-1;

mu_off  = nan(numel(off_vec),4);
sg_off  = nan(numel(off_vec),4);
Ns_off  = zeros(numel(off_vec),4);
metric  = -inf(numel(off_vec),1);   % métrica para escoger el mejor offset

for ii = 1:numel(off_vec)
    off = off_vec(ii);

    samp_orig = (0:NSyms-1)*UI + off + 1;     % índices de muestreo por símbolo (ORIGINAL)
    [tf, loc] = ismember(samp_orig, orig_idx_surv);

    sym_tx_i = sym(tf);                 % símbolos TX válidos (0..3)
    P_samp_i = P_out_trim(loc(tf));     % muestras de potencia (W) en ese offset

    if numel(sym_tx_i) < 20
        continue;   % offset inválido (muy pocos símbolos tras recortes)
    end

    P_s_mW_i = 1e3*P_samp_i(:);

    % medias y sigmas por nivel
    for s = 0:3
        m = (sym_tx_i == s);
        Ns_off(ii,s+1) = sum(m);
        if Ns_off(ii,s+1) > 0
            mu_off(ii,s+1) = mean(P_s_mW_i(m));
            sg_off(ii,s+1) = std(P_s_mW_i(m));
        end
    end

    % Ordenar por amplitud (por si no es estrictamente monótono)
    [mu_s_i, ord_i] = sort(mu_off(ii,:),'ascend');
    sg_s_i = sg_off(ii,ord_i);

    if any(isnan(mu_s_i)) || any(isnan(sg_s_i)) || any(sg_s_i<=0)
        continue;
    end

    % Métrica robusta: maximizar el "peor" Q entre niveles adyacentes
    % Q_ij ≈ (mu_j - mu_i)/(sigma_i + sigma_j)
    Q12 = (mu_s_i(2)-mu_s_i(1)) / (sg_s_i(1)+sg_s_i(2));
    Q23 = (mu_s_i(3)-mu_s_i(2)) / (sg_s_i(2)+sg_s_i(3));
    Q34 = (mu_s_i(4)-mu_s_i(3)) / (sg_s_i(3)+sg_s_i(4));

    metric(ii) = min([Q12 Q23 Q34]);
end

[best_metric, best_idx] = max(metric);

if ~isfinite(best_metric)
    error('No se pudo encontrar un offset válido. Revisa recortes/transitorio o aumenta NBits.');
end

off_best = off_vec(best_idx);

fprintf('\nMejor offset de decisión: off_best = %d muestras (%.4f UI) | métrica=%.4f\n', ...
    off_best, off_best/UI, best_metric);

% --- Con el mejor offset, volver a muestrear y calcular BER/SER ---
samp_orig = (0:NSyms-1)*UI + off_best + 1;
[tf, loc] = ismember(samp_orig, orig_idx_surv);

sym_tx = sym(tf);                 % símbolos transmitidos (0..3) válidos
P_samp = P_out_trim(loc(tf));     % potencia muestreada (W) en el mejor offset

if numel(sym_tx) < 20
    error('Muy pocos símbolos válidos para estimar SER/BER. Aumenta NBits o reduce recortes.');
end

P_s_mW = 1e3*P_samp(:);   % [mW]

m0 = (sym_tx==0); m1 = (sym_tx==1); m2 = (sym_tx==2); m3 = (sym_tx==3);
P0 = P_s_mW(m0); P1 = P_s_mW(m1); P2 = P_s_mW(m2); P3 = P_s_mW(m3);

mu = [mean(P0) mean(P1) mean(P2) mean(P3)].';          % [mW]
sg = [std(P0)  std(P1)  std(P2)  std(P3)].';           % [mW]

% Ordenar por amplitud (por si el SOA no queda estrictamente monótono)
[mu_s, order_est] = sort(mu, 'ascend');
sg_s = sg(order_est);

% Umbrales (óptimos para gaussianas con varianzas distintas)
Th1 = (sg_s(1)*mu_s(2) + sg_s(2)*mu_s(1)) / (sg_s(1) + sg_s(2));
Th2 = (sg_s(2)*mu_s(3) + sg_s(3)*mu_s(2)) / (sg_s(2) + sg_s(3));
Th3 = (sg_s(3)*mu_s(4) + sg_s(4)*mu_s(3)) / (sg_s(3) + sg_s(4));

% Histogramas
figure(2); clf; hold on; grid on;
histogram(P0,100);
histogram(P1,100);
histogram(P2,100);
histogram(P3,100);

yl = ylim;
plot([Th1 Th1], yl, 'r--', 'LineWidth', 1.2);
plot([Th2 Th2], yl, 'c--', 'LineWidth', 1.2);
plot([Th3 Th3], yl, 'b--', 'LineWidth', 1.2);

xlabel(sprintf('P_s [mW] (muestreo en off=%d muestras = %.3f UI)', off_best, off_best/UI));
ylabel('Counts');
title('Histogramas de P_{out} por nivel PAM-4 + umbrales (mejor offset de decisión)');
hold off;

% Detección por umbrales
sym_hat_sorted = zeros(size(P_s_mW));
sym_hat_sorted(P_s_mW < Th1) = 1;
sym_hat_sorted(P_s_mW >= Th1 & P_s_mW < Th2) = 2;
sym_hat_sorted(P_s_mW >= Th2 & P_s_mW < Th3) = 3;
sym_hat_sorted(P_s_mW >= Th3) = 4;

% Regresar a etiquetas 0..3 (invirtiendo el ordenamiento)
sym_hat = order_est(sym_hat_sorted) - 1;

% SER
SER = mean(sym_hat(:) ~= sym_tx(:));
fprintf('\nSER = %.3e (errores=%d de %d símbolos)\n', ...
    SER, sum(sym_hat(:) ~= sym_tx(:)), numel(sym_tx));

% BER (Gray PAM-4: 0->00, 1->01, 2->11, 3->10)
sym2bits = @(s) [ ...
    (s==2 | s==3); ...  % b2 (MSB)
    (s==1 | s==2)  ...  % b1 (LSB)
];

bits_tx  = zeros(2*numel(sym_tx),1);
bits_hat = zeros(2*numel(sym_hat),1);

for k = 1:numel(sym_tx)
    bits_tx(2*k-1:2*k)  = sym2bits(sym_tx(k));
    bits_hat(2*k-1:2*k) = sym2bits(sym_hat(k));
end

BER = mean(bits_hat ~= bits_tx);
fprintf('BER = %.3e (errores=%d de %d bits)\n', ...
    BER, sum(bits_hat ~= bits_tx), numel(bits_tx));

%% Eye diagrams ya alineados y sin transitorio
EyeSpanUI = 4;                  % ventana en UI
n     = EyeSpanUI * UI;         % samples per span
period = EyeSpanUI;             % eje x en UI
offset = floor(UI/2);           % centrar ojo en medio del símbolo

% --- Eye: I [A]
eyediagram(I_trim(:), n, period, offset);
ax = gca;
yl = prctile(I_trim(:), [0.5 99.5]);
pad = 0.30 * diff(yl);
ylim(ax, [yl(1)-pad, yl(2)+pad]);
title('Eye Diagram - Driving Current I (aligned, no transient)');
xlabel('Time (UI)'); ylabel('I [A]');

% --- Eye: P_out [mW]
eyediagram(1e3*P_out_trim(:), n, period, offset);
ax = gca;
yl = prctile(1e3*P_out_trim(:), [0.5 99.5]);
pad = 0.30 * diff(yl);
ylim(ax, [yl(1)-pad, yl(2)+pad]);
title('Eye Diagram - Output Power P_{out} (aligned, no transient)');
xlabel('Time (UI)'); ylabel('P_{out} [mW]');

% Línea del instante de decisión (en el eje del diagrama de ojo)
% Nota: eyediagram está centrado en offset=floor(UI/2), por eso se corrige así:
x_dec = (off_best - floor(UI/2)) / UI;   % [UI]
xline(x_dec,'r-','Decision', ...
    'LineWidth',1.5, ...
    'LabelVerticalAlignment','bottom', ...
    'LabelHorizontalAlignment','center');

drawnow;

elapsed = toc(tTotal);
fprintf('\nTiempo TOTAL de ejecucion: %.3f s (%.3f min)\n', elapsed, elapsed/60);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Funciones auxiliares (al final del archivo)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function N = SOA_N_RK4_uniform(TimeVec, I, Pseg, Ninit, GeomParams, MatParams, segment_length)
% Integra N(t) para UN segmento usando RK4 explícito en malla uniforme.
% Para t+dt/2 usa promedio discreto.

Nt = length(TimeVec);
N  = zeros(Nt,1);
N(1) = Ninit;

dt = TimeVec(2) - TimeVec(1);

q    = 1.602176634e-19; % C
Ener = 6.62607015e-34 * 299792458 / MatParams.lambda; % J (hf)

Vol_seg = GeomParams.width * GeomParams.depth * segment_length;

for k = 1:Nt-1
    II_k   = I(k);
    II_k1  = I(k+1);
    II_mid = 0.5*(II_k + II_k1);

    PP_k   = Pseg(k);
    PP_k1  = Pseg(k+1);
    PP_mid = 0.5*(PP_k + PP_k1);

    y = N(k);

    k1 = soa_dNdt(y,              II_k,   PP_k,   q, Ener, Vol_seg, GeomParams, MatParams, segment_length);
    k2 = soa_dNdt(y + 0.5*dt*k1,  II_mid, PP_mid, q, Ener, Vol_seg, GeomParams, MatParams, segment_length);
    k3 = soa_dNdt(y + 0.5*dt*k2,  II_mid, PP_mid, q, Ener, Vol_seg, GeomParams, MatParams, segment_length);
    k4 = soa_dNdt(y + dt*k3,      II_k1,  PP_k1,  q, Ener, Vol_seg, GeomParams, MatParams, segment_length);

    N(k+1) = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);

end

end

function dNdt = soa_dNdt(N, II, PP, q, Ener, Vol_seg, GeomParams, MatParams, segment_length)
% RHS de N(t) en un segmento

RR = MatParams.A*N + MatParams.B*(N.^2) + MatParams.C*(N.^3);

% Emisión estimulada
Tercero = GeomParams.Gamma * MatParams.DiffGain * (N - MatParams.N0) .* PP .* segment_length ./ (Vol_seg * Ener);

dNdt = II/(q*GeomParams.Vol) - RR - Tercero;

end
