# 1) Toy 2D
#python scripts/run_toy.py --dataset moons --noise 0.0 --n-samples 2000 --protos 24 --seed 7 --out runs/toy.png

# 2) CIFAR-10
#python scripts/run_cifar.py --out runs/cifar10.csv
#python scripts/plot_memcurve_bars.py --csvs runs/cifar10.csv --out runs/summary.png

# 3) IDC (HF)
python scripts/run_idc.py
# -> runs/idc_tsne_joint_proto.png  and  runs/idc_graphmemory.html
