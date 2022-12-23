%clear all
%close all
%clc

%%
SGpoly = 3;
SGframe = 9;

wlow = 500; %lower wavenumber limit you want to use 
whigh = 3500; % higher wavenumber limit you want to use
filetype = 'renishaw';
%change to 'cora', 'renishaw', or 'csv' depending on type of scan


%%% Choice of nf depends on the shape/frequency of baseline; linear baseline usually
% requires nf=2, more variable baseline requires higher value of nf.

% Value of p depends on the noise level of raw signal vector; Reletivly lower p
% value require for signal with high noise or higher value for vice versa

wspace = whigh-wlow+1;
wave = linspace(wlow,whigh,wspace);
wave_numb = wave';

nf = 2;
p = 0.009;


%select correction by changing correction to either baseline or derivative.

baseline = 'baseline';
derivative = 'derivative';

correction = 'baseline';



%Image/Plot Generation Selection

Image = '3D';
Location_Classify = 'Location';


%% generate spectra 1

folder = dir('C:\Users\Student003\Downloads\(2) PCB HER2 + CA 15-3\SERS\Interaction\2+'); % folder name
Spec = 'C:\Users\Student003\Downloads\(2) PCB HER2 + CA 15-3\SERS\Interaction\2+'; % folder name
mds1 = 1; %counter used

fsize = size(folder,1) - 2;
n=0;
Mapped_ds1 = zeros(whigh-wlow+1,fsize); %prealloate matrix to place spectra into
for n= 1:fsize % number of spectra collected
        
        filename = folder(n+2).name;
        if strcmp (filetype,'renishaw')
            delimiter = '	';        
            A = importdata(fullfile(Spec,filename), delimiter, 14).data;
        %A = flip(A,1);
        %A = csvread(fullfile(Spec,filename),1,0); % Alternative method.
        
            x0 = A(:,1);
            y0 = sgolayfilt(A(:,2),SGpoly,SGframe); %filters spectra
            
        elseif strcmp (filetype,'cora')
            
            delimiter = ',';
            A = importdata(fullfile(Spec,filename), delimiter, 1);
            x0 = A(:,1);
            y0 = sgolayfilt(A(:,2),SGpoly,SGframe); %filters spectra
            
        elseif strcmp (filetype,'csv')
            delimiter = ',';
            A = csvread(fullfile(Spec,filename),1,0); % Alternative method.
            x0 = A(:,1);
            y0 = sgolayfilt(A(:,2),SGpoly,SGframe); %filters spectra
        end
        
        %if statement determines if baseline correction or derivative
        %correction performed
        
        if strcmp(correction, baseline)
            [y_zint] = SGfilter_Baseline_Int(x0,y0);

       elseif strcmp(correction, derivative)
            [y_zint] = SGfilter_Derivative(y0, x0);
        end
        
        %
        %[yzint]= Int(x0,wave_numb, y_zint) ;  
        [yzint]= Int(x0,wave_numb, y0) ;
        Mapped_ds1(:,mds1) = yzint(:);
        mds1 = mds1+1;
   
end   
 
if fsize == 1
    avgmap1 = Mapped_ds1.';
else
    avgmap1 = mean(Mapped_ds1.');
end


%% generate spectra 2

folder = dir('C:\Users\Student003\Downloads\(2) PCB HER2 + CA 15-3\SERS\Interaction\3+'); % folder name
Spec = 'C:\Users\Student003\Downloads\(2) PCB HER2 + CA 15-3\SERS\Interaction\3+'; % folder name
mds1 = 1; %counter used

fsize = size(folder,1) - 2;
n=0;
Mapped_ds2 = zeros(whigh-wlow+1,fsize); %prealloate matrix to place spectra into
for n= 1:fsize % number of spectra collected
        
        filename = folder(n+2).name;
        if strcmp (filetype,'renishaw')
            delimiter = '	';        
            A = importdata(fullfile(Spec,filename), delimiter, 14).data;
        %A = flip(A,1);
        %A = csvread(fullfile(Spec,filename),1,0); % Alternative method.
        
            x0 = A(:,1);
            y0 = sgolayfilt(A(:,2),SGpoly,SGframe); %filters spectra
            
        elseif strcmp (filetype,'cora')
            
            delimiter = ',';
            A = importdata(fullfile(Spec,filename), delimiter, 1);
            x0 = A(:,1);
            y0 = sgolayfilt(A(:,2),SGpoly,SGframe); %filters spectra
            
        elseif strcmp (filetype,'csv')
            delimiter = ',';
            A = csvread(fullfile(Spec,filename),1,0); % Alternative method.
            x0 = A(:,1);
            y0 = sgolayfilt(A(:,2),SGpoly,SGframe); %filters spectra
        end
        
        %if statement determines if baseline correction or derivative
        %correction performed
        
        if strcmp(correction, baseline)
            [y_zint] = SGfilter_Baseline_Int(x0,y0);

       elseif strcmp(correction, derivative)
            [y_zint] = SGfilter_Derivative(y0, x0);
        end
        
        %
        [yzint]= Int(x0,wave_numb, y_zint) ;  
        Mapped_ds2(:,mds1) = yzint(:);
        mds1 = mds1+1;
   
end   
 
if fsize == 1
    avgmap2 = Mapped_ds2.';
else
    avgmap2 = mean(Mapped_ds2.');
end


%}
%% generate spectra 3

folder = dir('C:\Users\Student003\Downloads\(2) PCB HER2 + CA 15-3\SERS\Interaction\14 UmL'); % folder name
Spec = 'C:\Users\Student003\Downloads\(2) PCB HER2 + CA 15-3\SERS\Interaction\14 UmL'; % folder name
mds1 = 1; %counter used

fsize = size(folder,1) - 2;
n=0;
Mapped_ds3 = zeros(whigh-wlow+1,fsize); %prealloate matrix to place spectra into
for n= 1:fsize % number of spectra collected
        
        filename = folder(n+2).name;
        if strcmp (filetype,'renishaw')
            delimiter = '	';        
            A = importdata(fullfile(Spec,filename), delimiter, 14).data;
        %A = flip(A,1);
        %A = csvread(fullfile(Spec,filename),1,0); % Alternative method.
        
            x0 = A(:,1);
            y0 = sgolayfilt(A(:,2),SGpoly,SGframe); %filters spectra
            
        elseif strcmp (filetype,'cora')
            
            delimiter = ',';
            A = importdata(fullfile(Spec,filename), delimiter, 1);
            x0 = A(:,1);
            y0 = sgolayfilt(A(:,2),SGpoly,SGframe); %filters spectra
            
        elseif strcmp (filetype,'csv')
            delimiter = ',';
            A = csvread(fullfile(Spec,filename),1,0); % Alternative method.
            x0 = A(:,1);
            y0 = sgolayfilt(A(:,2),SGpoly,SGframe); %filters spectra
        end
        
        %if statement determines if baseline correction or derivative
        %correction performed
        
        if strcmp(correction, baseline)
            [y_zint] = SGfilter_Baseline_Int(x0,y0);

       elseif strcmp(correction, derivative)
            [y_zint] = SGfilter_Derivative(y0, x0);
        end
        
        %
        [yzint]= Int(x0,wave_numb, y_zint) ;  
        Mapped_ds3(:,mds1) = yzint(:);
        mds1 = mds1+1;
   
end   
 
if fsize == 1
    avgmap3 = Mapped_ds3.';
else
    avgmap3 = mean(Mapped_ds3.');
end

%% generate spectra 4

folder = dir('C:\Users\Student003\Downloads\(2) PCB HER2 + CA 15-3\SERS\Interaction\POS'); % folder name
Spec = 'C:\Users\Student003\Downloads\(2) PCB HER2 + CA 15-3\SERS\Interaction\POS'; % folder name
mds1 = 1; %counter used

fsize = size(folder,1) - 2;
n=0;
Mapped_ds4 = zeros(whigh-wlow+1,fsize); %prealloate matrix to place spectra into
for n= 1:fsize % number of spectra collected
        
        filename = folder(n+2).name;
        if strcmp (filetype,'renishaw')
            delimiter = '	';        
            A = importdata(fullfile(Spec,filename), delimiter, 14).data;
        %A = flip(A,1);
        %A = csvread(fullfile(Spec,filename),1,0); % Alternative method.
        
            x0 = A(:,1);
            y0 = sgolayfilt(A(:,2),SGpoly,SGframe); %filters spectra
            
        elseif strcmp (filetype,'cora')
            
            delimiter = ',';
            A = importdata(fullfile(Spec,filename), delimiter, 1);
            x0 = A(:,1);
            y0 = sgolayfilt(A(:,2),SGpoly,SGframe); %filters spectra
            
        elseif strcmp (filetype,'csv')
            delimiter = ',';
            A = csvread(fullfile(Spec,filename),1,0); % Alternative method.
            x0 = A(:,1);
            y0 = sgolayfilt(A(:,2),SGpoly,SGframe); %filters spectra
        end
        
        %if statement determines if baseline correction or derivative
        %correction performed
        
        if strcmp(correction, baseline)
            [y_zint] = SGfilter_Baseline_Int(x0,y0);

       elseif strcmp(correction, derivative)
            [y_zint] = SGfilter_Derivative(y0, x0);
        end
        
        %
        [yzint]= Int(x0,wave_numb, y_zint) ;  
        Mapped_ds4(:,mds1) = yzint(:);
        mds1 = mds1+1;
   
end   
 
if fsize == 1
    avgmap4 = Mapped_ds4.';
else
    avgmap4 = mean(Mapped_ds4.');
end



 %%
 
    blue_c  = [8 180 238] ./ 255; %blue
    gold_c = [247 198 9] ./ 255; %gold
    green_c = [57 187 84] ./ 255;5; %green
    purple_c = [148 29 184] ./ 255; %purple
    red_c = [230 18 9] ./ 255; %red
    orange_c = [1 0.5 0] ; %orange
 
    
offset = 4000;
offset1 = 1*offset;
offset2 = 2*offset;
offset3 = 3*offset;
offset4 = 4*offset;
offset5 = 5*offset;
offset6 = 6*offset;
offset7 = 7*offset;

%
FS =24; %font size

figure
 p = plot(wave_numb,avgmap1,wave_numb,avgmap2+50,wave_numb,avgmap3+100,wave_numb,avgmap4+150,...
        'LineWidth',1.1);
       hold on
       
    xlabel('Raman Shift (cm^-^1)','FontSize', 12, 'FontName','Arial')
    %ylabel('Normalized Distance','FontSize', 12, 'FontName','Arial')
    ylabel('Intensity (Au)','FontSize', 12, 'FontName','Arial')
    %title('Original PCBs')
    legend({'2+','3+','14 U/mL','POS'},'FontSize', FS, 'FontName','Arial')
    legend('Location', 'NorthEast')
%    xticks([500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800])
%    xticklabels([500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800])

%figure
% p = plot(wave_numb,avgmap2,...
%        'LineWidth',1.1);
       
%    xlabel('Raman Shift (cm^-^1)','FontSize', 12, 'FontName','Arial')
%    %ylabel('Normalized Distance','FontSize', 12, 'FontName','Arial')
%    ylabel('Intensity (Au)','FontSize', 12, 'FontName','Arial')
%    title('Sample 2')
%    legend({'Sample 1'},'FontSize', FS, 'FontName','Arial')
%    legend('Location', 'NorthEast')
    %ax.TickDir = 'out';
   % set(gca,'box','off', 'color', 'none')
    %b = axes('Position',get(ax,'Position'),'box','on','xtick',[],'ytick',[]);
    %b = axes('box','on','xtick',[],'ytick',[]);

    %b.LineWidth = 2;
    %axes(ax)
    %linkaxes([ax b], 'position')
 %}
 