import sys, runpy, torch
free, total = torch.cuda.mem_get_info()
frac = free / total
print(f"Capping PyTorch at {free/2**30:.2f} GiB of {total/2**30:.2f} GiB (fraction {frac:.3f}) to block sysmem fallback")
torch.cuda.set_per_process_memory_fraction(frac)
sys.path.insert(0, "part_d_attention"); sys.path.insert(0, ".")
sys.argv = ["part_d_attention/benchmark_attention.py"] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
