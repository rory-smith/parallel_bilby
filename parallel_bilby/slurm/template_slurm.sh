#!/bin/bash
#SBATCH --job-name={{job_name}}
#SBATCH --nodes={{nodes}}
#SBATCH --ntasks-per-node={{ntasks_per_node}}
#SBATCH --time={{time}}
#SBATCH --output={{log_file}}
#SBATCH --mem-per-cpu={{mem_per_cpu}}
{{slurm_extra_lines}}

{{bash_extra_lines}}

{{command}}
