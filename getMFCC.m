files_d = '/Users/Ahmed/Downloads/QSIURP2016/files/';
fldrs = dir(files_d);
% display(fldrs)
for j = 3:numel(fldrs)
    if (isdir(strcat(files_d,fldrs(j).name)) && numel(strfind(fldrs(j).name,'_R')) == 0)      
        files = dir(strcat(files_d,fldrs(j).name));
        for i = 3:numel(files)
            fname = files(i).name;
            display(fname);
            if (fname(length(fname)-3:length(fname)) == '.wav')
                fpath = (strcat(files_d,fldrs(j).name,'/',fname));
                display(fpath)
                [y,Fs] = audioread(fpath);
                hamming = @(N)(0.54-0.46*cos(2*pi*[0:N-1].'/(N-1)));
                [CCs,FBEs,frames] = mfcc(y,Fs,25,10,0.97,hamming,[50 15000],20,20,22);
                dlmwrite(strcat(fpath,'.mfcc'),transpose(CCs));
                dlmwrite(strcat(fpath,'.fbe'),transpose(FBEs));
            end
        end
    end
end