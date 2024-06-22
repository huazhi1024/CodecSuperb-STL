#coding=utf8
import os
import dac
from audiotools import AudioSignal

# Set CUDA environment variable to use the first GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = '3' ##在linux终端才有用



# 前面先debug下采样和上采样音频的正确性
def read_a_signal(wavpath,target_sample_rate):
    signal = AudioSignal(wavpath)
    original_sample_rate = signal.sample_rate
    if original_sample_rate != target_sample_rate: ## 检查是否需要重采样
        signal.resample(target_sample_rate) # 重采样音频
    return signal,original_sample_rate,target_sample_rate

def save_a_signal(signal,original_sample_rate,wavpath_out):
    if signal.sample_rate!=original_sample_rate:
        signal.resample(original_sample_rate) # 还原为原始采样率
        signal.write(wavpath_out)
    else:
        signal.write(wavpath_out)



################################################
model_dict={
"16khz":{
    "model":"/home/huiyu/workspace/Codec/DAC/descript-audio-codec-main/modelres_bak_pretrain/16khz_320d4q_2kbps/best/dac/weights.pth",
    "target_sample_rate":16000},

"44.1khz":{
    "model":"/home/huiyu/workspace/Codec/DAC/descript-audio-codec-main/modelres_bak_pretrain/44khz_512d8q_7kbps/best/dac/weights.pth",
    "target_sample_rate":44100},

"48khz":{
    "model":"/home/huiyu/workspace/Codec/DAC/descript-audio-codec-main/modelres_bak_pretrain/48khz_512d8q_7.5kbps/best/dac/weights.pth",
    "target_sample_rate":48000},

}
#################################################################################################################################################################
data_16khz=["fluent_speech_commands","libri2Mix_test","librispeech","quesst","snips_test_valid_subset","voxceleb1","RAVDESS","vox1_test_wav","LibriSpeech"]
data_44khz=["ESC-50-master"]
data_48khz=["crema_d","esc50","fsd50k","gunshot_triangulation","vox_lingua_top10"]
data_list=data_16khz+data_44khz+data_48khz
####################################################################################################################################################################

#############################
tag="16khz" #change
target_sample_rate =model_dict[tag]["target_sample_rate"]  
model_path=model_dict[tag]["model"] 
output_dirname = f"codecsuperb_dac_{tag}" 
##################################################################
# Define the model path and load the DAC model
model = dac.DAC.load(model_path)
model.cuda()
print(f"processing {tag}")
#######################################



#########################OK###############################################
for name in data_list:
    ##############################
    print(f"processing {name}")
    scpfile= f"/home/huiyu/workspace/Codec/Codec-SUPERB-SLT_Challenge/Codec-SUPERB-SLT_Challenge/codec_superb_data/ref_data/scp/{name}.wav.scp"
    wavlist=[]
    with open(scpfile,'r',encoding='utf8')as fr:
        for line in fr:
            line=line.strip()
            wavname,wavpath=line.split()
            wavlist.append(wavpath)
    #######################################
    # Loop through all audio files in the input directory
    for filename in wavlist:
        #####################################
        output_file=filename.replace("ref_data",f"{output_dirname}")
        output_root=os.path.dirname(output_file)
        if not os.path.exists(output_root):
            os.makedirs(output_root,exist_ok=True)
        ##########################
        # Load the audio signal and move it to the GPU
        signal,original_sample_rate,target_sample_rate=read_a_signal(wavpath=filename,target_sample_rate=target_sample_rate)
        signal.cuda()
        #method1
        # Compress and decompress the signal
        try:
            x = model.compress(signal)
            y = model.decompress(x)
            y = y.detach().cpu()
            #############################
            #method2-容易OOM
            ####################################
            # Write the decompressed audio to the output file
            save_a_signal(y,original_sample_rate,output_file)
        except:
            print(f"{filename} error")
    #############################################
    print(f"processing {tag}--{name} done")
#################################################################
