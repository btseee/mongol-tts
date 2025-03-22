import torch

def clean_gpu_cache():
    """
    Хэрэв GPU байгаа бол GPU-ийн кэшийг цэвэрлэж, дээд санах ойг дахин тохируулна.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # GPU-ийн кэшийг цэвэрлэх
        torch.cuda.reset_peak_memory_stats()  # Дээд санах ойн статистикийг дахин тохируулах
    else:
        print("GPU илрэхгүй байна, кэш цэвэрлэхийг алгаслаа.")
        
# Тохиргоог турших
if __name__ == "__main__":
    clean_gpu_cache()
# Энэ код нь GPU-н кэшээ цэвэрлэх функц болон туршилтын кодыг агуулсан байна. Энэ код нь GPU-н кэшээ цэвэрлэх функцыг ажиллуулахад бэлэн байна уу?
