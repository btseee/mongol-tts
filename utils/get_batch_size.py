import torch
import os

# GPU санах ойд суурилсан багц хэмжээ тооцоолох функц
def get_auto_batch_size(is_training=True, min_batch=2, max_batch=64):
    """
    Багц хэмжээг автомат тооцоолох функц нь ашиглах боломжтой GPU санах ойд суурилна.
    
    Аргументууд:
        is_training (bool): Хэрэв True бол сургалт хийхэд оптимизаци хийх; хэрэв False бол үнэлгээ хийхэд.
        min_batch (int): Үйлдэл ажиллагаа хэвийн байхын тулд хамгийн бага багц хэмжээ.
        max_batch (int): Санах ойд дарамт учруулахгүйгээр хамгийн их багц хэмжээ.
    
    Буцаах:
        int: Тооцоолсон багц хэмжээ.
    """
    if not torch.cuda.is_available():
        return min_batch  # Хэрэв GPU ашиглах боломжгүй бол хамгийн бага хэмжээг авах

    # Нийт GPU санах ой ба одоогийн хувиарлагдсан санах ойн хэмжээ авах
    gpu_memory = torch.cuda.get_device_properties(0).total_memory  # байт-аар
    available_memory = torch.cuda.memory_allocated(0)  # байт-аар
    free_memory = gpu_memory - available_memory  # байт-аар

    # Сургалт эсвэл үнэлгээ хийхэд үндэслэн хүчин зүйл тохируулах
    factor = 0.7 if is_training else 0.85  # Илүү их GPU санах ой ашиглах

    # Төлбөргүй санах ойг GB болгож хөрвүүлж, багц хэмжээ тооцоолно
    estimated_batch_size = int((free_memory / (1024 ** 3)) * factor)
    
    # Багц хэмжээг хамгийн бага ба хамгийн их хооронд хязгаарлана
    return max(min_batch, min(estimated_batch_size, max_batch))

# CPU цөмийн тоо дээр үндэслэн өгөгдлийн ачаалал өгөгдлийн удирдагчдын тоог тодорхойлох функц
def get_auto_num_workers(is_training=True):
    """
    CPU цөмийн тоонд суурилсан өгөгдлийн ачаалал удирдагчдын тоог автомат тохируулна.
    
    Аргументууд:
        is_training (bool): Хэрэв True бол сургалт хийхэд оптимизаци хийх; хэрэв False бол үнэлгээ хийхэд.
    
    Буцаах:
        int: Ачаалал удирдагчдын тоо.
    """
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 4  # Хэрэв CPU тоо байхгүй бол 4 гэж тохируулах

    # Илүү их CPU цөмийг ашиглах: сургалтанд 16 хүртэл, үнэлгээнд 8 хүртэл
    return min(16, cpu_count) if is_training else min(8, cpu_count // 2)

# Тохиргоо хийх жишээ
def setup_config():
    """
    Автомат тооцоолсон параметрүүдтэй тохиргоог тохируулна.
    
    Буцаах:
        dict: Багц хэмжээ болон ажилчдын тоо бүхий тохиргоо.
    """
    # Параметрүүдийг тооцоолно
    batch_size = get_auto_batch_size(is_training=True, max_batch=64)
    eval_batch_size = get_auto_batch_size(is_training=False, max_batch=128)  # Үнэлгээнд томруулах
    num_loader_workers = get_auto_num_workers(is_training=True)
    num_eval_loader_workers = get_auto_num_workers(is_training=False)

    # Тохиргооны жишээ үгсийн сан
    config = {
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "num_loader_workers": num_loader_workers,
        "num_eval_loader_workers": num_eval_loader_workers
    }
    
    return config

# Тохиргоог турших
if __name__ == "__main__":
    config = setup_config()
    print("Тохиргоо:")
    print(f"  batch_size: {config['batch_size']}")
    print(f"  eval_batch_size: {config['eval_batch_size']}")
    print(f"  num_loader_workers: {config['num_loader_workers']}")
    print(f"  num_eval_loader_workers: {config['num_eval_loader_workers']}")