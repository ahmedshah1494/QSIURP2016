function [out,ap,av,am] = gmmMAPadapt(x,mmn,mvr,mwt,relv,varOrn,scaledsup)

%This implmentation also considers the scaling factors for supervectors as
%described in [1][2]. The idea is to compare two distributions using KL
%divergence. It is then shown that it is upperbounded by a linear kernel.
%To define this linear kernel scaling using the covariance matrix is done.
%Note-  They assume that the variances are not adapated. Hence the UBM
%variances are used. For diagonal covariance this simply implies scaling by
%sqaure root of variance in each dimension

%This version simply assumes that mean,wts and covariances are given.
%

%gmmMAPAdapt(x,mmn,mvr,mwt,relevance);
%gmmMAPadapt(x,relevance,gmmixt)
%x -- t X d dimensional
%mmn -- nComp X d dimensional
%mvn -- assumes diagonal covariance only -- nComp X d dimensional
%wtmn -- 1 X nComp wt of each mixture
%relv - relevance factor (between 0 to 1) -- see below
%varOrn - include variance or not (1 or 0)
%scaledsup - scaling as descirbed above or not (1 or 0)

%setting relevance factor is a pain - this paper [4] says it is normally set
%between 8 to 20 

%currently setting it to relv(between 0 and 1)*times total number of vectors to be adapated
%this means  for relv=0.3, if 20%of vectors belong to component M_i then the new data will
%contribute 0.2/(0.2+0.3) = 0.4 i.e 40% to the updated means of that
%component

%[1] - Dehak, Réda, et al. "Linear and non linear kernel GMM supervector machines for speaker verification." INTERSPEECH. 2007.
%[2] - Campbell, William M., et al. "SVM based speaker verification using a GMM supervector kernel and NAP variability compensation." Acoustics, Speech and Signal Processing, 2006. ICASSP 2006 Proceedings. 2006 IEEE International Conference on. Vol. 1. IEEE, 2006.
%[3] - Ferras, Marc, Koichi Shinoda, and Sadaoki Furui. "Structural MAP adaptation in GMM-supervector based speaker recognition." Acoustics, Speech and Signal Processing (ICASSP), 2011 IEEE International Conference on. IEEE, 2011.
%[4] - Charbuillet, Christophe, Damien Tardieu, and Geoffroy Peeters. "Gmm-supervector for content based music similarity." International Conference on Digital Audio Effects, Paris, France. 2011.

rows=size(x,1);
%relevance=relv*rows;
relevance=20;
n=size(mmn,1);
d=size(mmn,2);
am=zeros(n,d);
av=zeros(n,d);
ap=zeros(1,n);
%val=1e-10;
%corrmat=repmat(val,d,1);
%mval=repmat(corrmat',n,1);
%mmn=mmn+mval;


p_cap=zeros(rows,n);
for i=1:n
    p_cap(:,i)=(mvnpdf(x,mmn(i,:),mvr(i,:))*mwt(1,i));
end
%p_cap=p_cap+1e-300;
sumal=sum(p_cap,2)+1e-100;
sumall=repmat(sumal,1,n);
p_cap = p_cap./sumall;
p_cap = p_cap';


n_factors = sum(p_cap,2);
x2v = x.*x;
for i=1:n
    n_factor = n_factors(i,1);
    expected_m = sum(diag(p_cap(i,:))*x,1)/(n_factor+eps);
    expected_v = sum(diag(p_cap(i,:))*x2v,1)/(n_factor+eps);
    
    %expected_v=expected_v+corrmat';
    alpha=n_factor/(n_factor+relevance);
    
    
    am(i,:)=alpha.*expected_m+(1-alpha).*mmn(i,:);
    av(i,:)=alpha.*expected_v+(1-alpha).*(mvr(i,:)+mmn(i,:).*mmn(i,:))-(am(i,:).*am(i,:));
    ap(1,i)=(alpha*n_factor/rows)+(1-alpha)*mwt(1,i);
    
    if scaledsup
        %diagonal covariances -- means simply divide by sqrt of mvr(i,:)
        %the scaling assumes cov were not updated -- hence use original
        amt=sqrt(mwt(1,i))*am(i,:);
        am(i,:) = amt./sqrt(mvr(i,:));
    end
end

ap=ap./sum(ap);
amtr=am'; %take transpose so that each comp is in column
supvec=amtr(:);


if varOrn == 0
    out = supvec' ;
else
    avtr = av';
    out = [supvec;avtr(:)]';
end

% if sgnrtNrm == 1
%     out = sign(out).*sqrt(abs(out));
% end
% 
% if l2nrm == 1
%     out = out/norm(out);
% end



end

