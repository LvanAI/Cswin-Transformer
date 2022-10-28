
from .cswin import CSWin_64_24322_small_224

def cswin_small_224(args):
    model = CSWin_64_24322_small_224(
                                num_classes= args.num_classes,
                                drop_rate= args.drop_rate,
                                drop_path_rate= args.drop_path)
    return model