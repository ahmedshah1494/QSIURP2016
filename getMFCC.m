files_d = '\\andrew.ad.cmu.edu\users\users8\mshah1\Desktop\QSIURP2016\files\1030\';
files = dir(files_d);
for i = 3:numel(files):
    s
for i = 3:numel(files)
    fname = files(i).name;
    display(fname);
    if (fname(length(fname)-3:length(fname)) == '.wav')
        display(strcat(d,fname));
        [y,Fs] = audioread(strcat(d,fname));
        [CCs,FBEs,frames] = mfcc(y,Fs,25,10,0.97,hamming(44100),[50 15000],20,13,22);
        dlmwrite(strcat(d,fname,'.mfcc'),transpose(CCs));
        dlmwrite(strcat(d,fname,'.fbe'),transpose(FBEs));
    end
ends