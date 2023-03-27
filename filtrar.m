function [eeg, eog] = filtrar(signal_in, fs)
Q = 15;
fn = 60;
wo = fn/(fs/2);
bw = wo/Q;
[b,a] = iirnotch(wo,bw);
signal_in = filter(b,a,signal_in);

fn = 120;
wo = fn/(fs/2);
bw = wo/Q;
[b,a] = iirnotch(wo,bw);
signal_in = filter(b,a,signal_in);

fn = 180;
wo = fn/(fs/2);
bw = wo/Q;
[b,a] = iirnotch(wo,bw);
signal_in = filter(b,a,signal_in);

f1 = 1;
f2 = 45;
order = 2;

[b,a] = butter(order,(2*f1)/fs,'high');
signal_in = filtfilt(b,a,signal_in);
[b2,a2] = butter(order,(2*f2)/fs);
signal_in = filtfilt(b2,a2,signal_in);

eeg = signal_in(:,1:17);
eog = signal_in(:,22);

[b2,a2] = butter(order,(2*10)/fs);
eog = filtfilt(b2,a2,eog);

end