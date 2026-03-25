function ElecField = CWLaser(NSamples,SamplePeriod)
% Generates the Elec Field from a CW laser with Average Power AvgPow
% and a half laser linewidth Df (in Hz), which is WGN-generated.
% "Despliega" controls whether the Spectrum is displayed as figure.
% Ramon Gutierrez-Castrejon. Feb/2004.
% Input: Number of Samples and Sample Period 

%clear all
%NSamples =512;
%SamplePeriod = 12.5E-12;

% Parameters
%
AvgPow = 10;                          %  Avg Power in  mW
Df =50E3;                   % Half laser linewidth in Hz
Despliega =  0;                     % 1 = YES, 0 = NO
%--------------------------------------------------------------------------
%
AvgPow = AvgPow/1000;                               % Power in Watts
FreqPer = 1/(NSamples*SamplePeriod);         % Sample period in Freq. for graphic purposes
%w = sqrt(-2*log(rand(NSamples,1))).*sin(2*pi*rand(NSamples,1))*(2*pi*Df);
w = randn(NSamples,1)*(2*pi*Df);                % White Gaussian Noise through randn
%figure
%plot( abs(fft(w)).^2 )
%figure
%hist(w,30)
phi = cumsum(w)*SamplePeriod;                 % Integration of frequency vector w to create the phase vector

ElecField = cos(phi).*sqrt(AvgPow) + 1i*sin(phi).*sqrt(AvgPow);

if (Despliega ~= 0)                                % Displays the spectum of ElecField
     Spec = fft(ElecField);
%   Spec = [ Spec( (NSamples/2+1):NSamples); Spec(1:NSamples/2) ];
     Spec = fftshift(Spec);
     x = [ (-1*NSamples/2):-1  0:(NSamples/2-1)  ]' *FreqPer*1e-6;
     figure
     set(gca,'FontSize',14,'FontWeight','bold','LineWidth',1.8,'TickLength',[0.025 0.050])
     %plot(x, abs(Spec).^2 *SamplePeriod/NSamples);
     plot(x, abs(Spec).^2 /(NSamples^2));
     title('Specrum of Laser Output');
     xlabel('Frequency [MHz]');
     ylabel('Power [Watts]')
     straux = sprintf('Delta f = %g',Df);
     legend(straux);
     legend('boxoff');
 end
 
