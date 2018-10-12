
import argparse
from RGBHallucinator import Hallucinator

if __name__ == '__main__' :
    parser =  argparse.ArgumentParser(description = "Mention the scale and GPU parameters")
    parser.add_argument('conf_file',type = str, help = " path to configuration file " )
    parser.add_argument('scale',type = int ,help = " What scale do you want to train it at ")
    parser.add_argument('gpu',type = int,help = " Which GPU do you want to train it in ")
    args= parser.parse_args()
    conf = args.conf_file
    scale=args.scale
    gpu  =args.gpu
    H = Hallucinator(conf,scale,gpu)
    H.train()
    H.testAll()


