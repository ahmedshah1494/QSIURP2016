function computeFVandSupWhole(infllist,nComp,tocomp,gmmfolder,outlist1)
% addpath(genpath('VLFEATPATH'));
%infllist - input file list
%nComp -number of Gaussian in GMM
%tocomp - set it to 'S'
%gmmfolder - path where gmms are stored -- without nComp
%outlist1 - output filelist

%for supvec -- to the output path -- supMn,supMnVr,supMnScaVr,supMnSca
%directories will be added just before the filename
%supMn - mean only without any scaling of adapated means

%only computing supMn
%option of saving in libsvm or not


if nargin == 5
    if strcmp(tocomp,'F')
        fprintf('Computing only Fisher Vectors \n');
        fvpath = outlis1;
        fv=1;
        sp=0;
    elseif strcmp(tocomp,'S')
        fprintf('Computing only Supervectors \n')
        suppath = outlist1;
        fv=0;
        sp=1;
    else
        error('Usage:<infllist>,<nComp>,<gmmfolder>,<tocomp>,<outlist1>,<outlist2>');
    end
elseif nargin == 6
    fprintf('Computing both Fisher and Supervector \n');
    fvpath = outlist1;
    suppath = outlist2;
    fv=1;
    sp=1;
end

%parameters here
relv=20; %set to be constant
stid = 1;
edid = 60;
normft=0;
libsfmt=0;
%varOrn=storing both with and without variance
if ischar(nComp)
    nComp=str2double(nComp);
end


mns = load(fullfile(gmmfolder,num2str(nComp),'means.txt'));
wts = load(fullfile(gmmfolder,num2str(nComp),'weights.txt'));
cvs = load(fullfile(gmmfolder,num2str(nComp),'covariances.txt'));

if normft
    aldmn = load(fullfile(gmmfolder,num2str(nComp),'0meanNorm.txt'));
    aldmn = aldmn(:)';
    aldst = load(fullfile(gmmfolder,num2str(nComp),'1stdNorm.txt'));
    aldst = aldst(:)';
end

outdim1 = 2*prod(size(mns));
outindcs1 = strsplit(num2str([0:outdim1-1]),' ');
outindcs1 = strcat(outindcs1,':');

outdim2 = prod(size(mns));
outindcs2 = strsplit(num2str([0:outdim2-1]),' ');
outindcs2 = strcat(outindcs2,':');


infls=importdata(infllist);
if exist('fvpath','var')
    fvoutfls=importdata(fvpath);
    if size(infls,1) ~= size(fvoutfls,1)
        error('Mismatch in length of input output error files');
    end
end
if exist('suppath','var')
    supoutfls=importdata(suppath);
    if size(infls,1) ~= size(supoutfls,1)
        error('Mismatch in length of input output error files');
    end
end

if (fv && ~sp)
    disp('redundant');
elseif (~fv && sp)
    for i=1:size(infls,1)
        if exist(infls{i,1},'file')
            cdata = load(infls{i,1});
            cdata = cdata(:,stid:edid);
            if normft == 1
                cdata = bsxfun(@minus,cdata,aldmn);
                cdata = bsxfun(@times,cdata,1./aldst);
            end
            
            supvec1 = gmmMAPadapt(cdata,mns,cvs,wts,relv,0,1);
            
            if any(isnan(supvec1(1:outdim2))) || any(isinf(supvec1(1:outdim2)))
                warning(sprintf('Escaping for %s due to Nan or Inf in adapated means for scaled versions\n',infls{i,1}));
            else %save only when all are available
                
                [drpx,flpx,flpxpt]=fileparts(supoutfls{i,1});
                if ~isdir(fullfile(drpx,['supMnSca' num2str(relv)]))
                    mkdir(fullfile(drpx,['supMnSca' num2str(relv)]));
                end
                
                
                if libsfmt == 1
                    spstr = strsplit(num2str(supvec1(1:outdim2)),' ');
                    ftwrt = strcat(outindcs2,spstr);
                    fttowrt=strjoin(ftwrt,' ');
                    fd=fopen(fullfile(drpx,['supMnSca' num2str(relv)],[flpx flpxpt]),'w');
                    fprintf(fd,fttowrt);
                    fclose(fd);
                else
                    dlmwrite(fullfile(drpx,['supMnSca' num2str(relv)],[flpx flpxpt]),supvec1(1:outdim2));
                end
                
            end
            
        else
            fprintf('%s Not Found !!\n',infls{i,1});
        end
    end
    
else
    disp('Redundant');  
end
end

