
"""
Snakefile for 

Author: Davis McCarthy
Affiliation: St Vincent's Institute of Medical Research and the University of Melbourne

Run: snakemake -s Snakefile_canopy --jobs 1000 --latency-wait 30 --cluster-config cluster.json --cluster 'bsub -J {cluster.name} -q {cluster.queue} -n {cluster.n} -R "rusage[mem={cluster.memory}]" -M {cluster.memory}  -o {cluster.output} -e {cluster.error}' --keep-going --rerun-incomplete

Davis McCarthy, 02 January 2019
"""

import glob
import os
from subprocess import run
import subprocess
import pandas as pd
import re
import h5py

shell.prefix("set -euo pipefail;")

donors = ['bima', 'bubh', 'ceik', 'ciwj', 'cuhk', 'deyz', 'diku', 'eipl', 'eofe', 'euts', \
        'fawm', 'feec', 'fiaj', 'fikt', 'garx', 'gesg', 'gifk', 'hehd', 'heja', 'hipn', 'ieki', \
        'jogf', 'joxm', 'kajh', 'kuco', 'laey', 'lexy', 'melw', 'miaj', 'naju', 'nusw', 'oaaz', \
        'oaqd', 'oicx', 'oilg', 'pamv', 'pelm', 'pipw', 'puie', 'qayj', 'qolg', 'qonc', 'rozh', \
        'rutc', 'sebz', 'sehl', 'sohd', 'tixi', 'toss', 'ualf', 'vabj', 'vass', 'vils', 'vuna', \
        'wahn', 'wetu', 'wigw', 'wopl', 'wuye', 'xugn', 'xuja', 'zihe', 'zoxy']
## too few variants for clonal analysis:
singlecell_donors_all = ['bima', 'bubh', 'ceik', 'ciwj', 'cuhk', 'deyz', 'diku',\
                         'eipl', 'eofe', 'euts', 'fawm', 'feec', 'fiaj', 'fikt',\
                          'garx', 'gesg', 'gifk', 'hehd', 'heja', 'hipn', 'ieki',\
                          'joxm', 'kajh', 'kuco', 'laey', 'lexy', 'melw',\
                          'miaj', 'naju', 'nusw', 'oaaz', 'oaqd', 'oilg',\
                          'pamv', 'pelm', 'pipw', 'puie', 'qayj', 'qolg', 'qonc',\
                          'rozh', 'rutc', 'sebz', 'sehl', 'sohd', 'toss', 'ualf',\
                          'vabj', 'vass', 'vils', 'vuna', 'wahn', 'wetu', 'wigw',\
                          'wopl', 'wuye', 'xugn', 'xuja', 'zihe', 'zoxy'] # 60 donors
## lenient variant filtering
## donors with <10 variants with coverage in at least one cell:
## bima, bubh, ceik, cuhk, deyz, diku, dons, eika, fiaj, gifk, hehd, jogf, kajh, lise, pamv, pelm, rutc, sebz, tolg, toss, tuju, vabj, wigw, wopl, wuye, xuja, zihe
## not enough QC-passing cells (<30): ciwj, eipl, eofe, miaj, oaqd, 
donors_lenient_all = ['euts', 'fawm', 'feec', 'fikt', \
    'garx', 'gesg', 'heja', 'hipn', 'ieki', 'joxm', 'kuco', 'laey', 'lexy', 'melw', \
    'naju', 'nusw', 'oaaz', 'oilg', 'pipw', 'puie', 'qayj', 'qolg', 'qonc', 'rozh', \
    'sehl', 'sohd', 'ualf', 'vass', 'vils', 'vuna', 'wahn', 'wetu', 'xugn', 'zoxy'] ## 34 donors  
## Canopy will not fit (variant clustering fails): melw, sohd
donors_lenient_cell_cov = ['euts', 'fawm', 'feec', 'fikt', 'garx', 'gesg', \
    'heja', 'hipn', 'ieki', 'joxm', 'kuco', 'laey', 'lexy', 'naju', 'nusw', \
    'oaaz', 'oilg', 'pipw', 'puie', 'qayj', 'qolg', 'qonc', 'rozh', 'sehl', \
    'ualf', 'vass', 'vils', 'vuna', 'wahn', 'wetu', 'xugn', 'zoxy'] ## 32 donors
## strict variant filtering
## donors with <10 variants with coverage in at least one cell:
## bima, bubh, ceik, ciwj, cuhk, deyz, diku, dons, eika, fiaj, gifk, hehd, jogf, kajh, lexy, lise, pamv, pelm, rutc, sebz, tolg, toss, tuju, vabj, vils, wigw, wopl, wuye, xuja, zihe
## not enough QC-passing cells (<30): eipl, eofe, melw, miaj, oaqd
donors_strict_all = ['euts', 'fawm', 'feec', 'fikt', 'garx', 'gesg', \
    'heja', 'hipn', 'ieki', 'joxm', 'kuco', 'laey', 'naju', 'nusw', \
    'oaaz', 'oilg', 'pipw', 'puie', 'qayj', 'qolg', 'qonc', 'rozh', 'sehl', \
    'sohd', 'ualf', 'vass', 'vuna', 'wahn', 'wetu', 'xugn', 'zoxy'] # 31 donors
## Canopy will not fit (variant clustering fails): kuco, sohd
donors_strict_cell_cov = ['euts', 'fawm', 'feec', 'fikt', 'garx', 'gesg', \
    'heja', 'hipn', 'ieki', 'joxm', 'laey', 'naju', 'nusw', \
    'oaaz', 'oilg', 'pipw', 'puie', 'qayj', 'qolg', 'qonc', 'rozh', 'sehl', \
    'ualf', 'vass', 'vuna', 'wahn', 'wetu', 'xugn', 'zoxy'] # 29 donors

sce_list = {}
sce_list['filt_lenient'] = {}
sce_list['filt_lenient']['all_filt_sites'] = expand(\
    'data/sces/sce_{donor}_with_clone_assignments.filt_lenient.all_filt_sites.rds',\
    donor = donors_lenient_all)
sce_list['filt_lenient']['cell_coverage_sites'] = expand(\
    'data/sces/sce_{donor}_with_clone_assignments.filt_lenient.cell_coverage_sites.rds',\
    donor = donors_lenient_cell_cov)
sce_list['filt_strict'] = {}
sce_list['filt_strict']['all_filt_sites'] = expand(\
    'data/sces/sce_{donor}_with_clone_assignments.filt_strict.all_filt_sites.rds',\
    donor = donors_strict_all)
sce_list['filt_strict']['cell_coverage_sites'] = expand(\
    'data/sces/sce_{donor}_with_clone_assignments.filt_strict.cell_coverage_sites.rds',\
    donor = donors_strict_cell_cov)
sces_flat = []
sces_flat.append(sce_list['filt_lenient']['all_filt_sites'])
sces_flat.append(sce_list['filt_lenient']['cell_coverage_sites'])
sces_flat.append(sce_list['filt_strict']['all_filt_sites'])
sces_flat.append(sce_list['filt_strict']['cell_coverage_sites'])
sces_flat = [filename for elem in sces_flat for filename in elem]

rule all:
    input:
        expand('data/exome-point-mutations/high-vs-low-exomes.v62.ft.filt_{strictness}-{donor}.txt.gz',\
            strictness = ['lenient', 'strict'], donor = singlecell_donors_all),
        expand('data/raw/mpileup/{donor}.mpileup.vcf{suffix}', \
            donor = singlecell_donors_all, suffix = ['.gz', '.gz.csi']),
        expand('data/sces/sce_{donor}_with_clone_assignments.{strictness}.{sites}.rds',\
            donor = singlecell_assign_donors, strictness = ['filt_strict', 'filt_lenient'],\
            sites = ['all_filt_sites', 'cell_coverage_sites']),
        expand('reports/de_pathway/de_pathway.{cells}.{strictness}.{sites}.html', \
             cells = ['unst_cells'], strictness = ['filt_strict', 'filt_lenient'],\
             sites = ['all_filt_sites', 'cell_coverage_sites']), # 'cell_coverage_sites'
        expand('reports/de_pathway/de_pathway.{cells}.cellcycle_analyses.{strictness}.{sites}.html', \
             cells = ['unst_cells'], strictness = ['filt_strict', 'filt_lenient'],\
             sites = ['all_filt_sites', 'cell_coverage_sites']), # 'cell_coverage_sites'
        expand('reports/de_pathway/de_pathway.{cells}.permutations.{strictness}.{sites}.html', \
             cells = ['unst_cells'], strictness = ['filt_strict', 'filt_lenient'],\
             sites = ['all_filt_sites', 'cell_coverage_sites']),  
        expand('data/exome-point-mutations/high-vs-low-exomes.v62.ft.alldonors-{strictness}.all_filt_sites.ped', \
            strictness = ['filt_strict', 'filt_lenient']),
        expand('data/exome-point-mutations/high-vs-low-exomes.v62.ft.alldonors-{strictness}.all_filt_sites.vcf', \
            strictness = ['filt_strict', 'filt_lenient']),
        expand('data/simulations/{donor}.simulate.rds', \
            donor = donors_lenient_cell_cov),
        expand('data/variance_components/donorVar/{donor}.var_part.var1.csv' \
            donor = donors_lenient_cell_cov)


rule run_varpart_per_donor:
    input:
        sce=lambda wildcards: sce_list['filt_lenient']['cell_coverage_sites']
    output:
        'data/variance_components/donorVar/{donor}.var_part.var1.csv'
    conda:
        "envs/myenv.yaml"
    singularity:
        "docker://davismcc/r-singlecell-img"
    shell:
        'Rscript src/R/var_part_donor.R {wildcards.donor}'


rule run_simulation_per_donor:
    input:
        card='data/cell_assignment/cardelino_results.{donor}.filt_lenient.cell_coverage_sites.rds'
    output:
        real_data='data/simulations/{donor}.filt_lenient.cell_coverage_sites.mult.rds',
        simu_data='data/simulations/{donor}.simulate.rds'
    conda:
        "envs/myenv.yaml"
    singularity:
        "docker://davismcc/r-singlecell-img"
    shell:
        'Rscript src/R/simulation_per_donor.R {wildcards.donor}'


rule run_de_pathway_analysis_unst_cells_permutation:
    input:
        sce=lambda wildcards: sce_list[wildcards.strictness][wildcards.sites]
    output:
        html='reports/de_pathway/de_pathway.unst_cells.permutations.{strictness}.{sites}.html',
        unst_rds='data/de_analysis_FTv62/permutations/{strictness}.{sites}.de_results_unstimulated_cells.rds'
    conda:
        "envs/myenv.yaml"
    singularity:
        "docker://davismcc/r-singlecell-img"
    shell:
        '{rscript_cmd} src/R/compile_report_de_pathways.R '
        '-c {wildcards.strictness}.{wildcards.sites} '
        '-o {output.html} '
        '--template src/Rmd/DE_pathways_FTv62_callset_clones_pairwise_vs_base.unst_cells.permutations.Rmd '
        '--title "DE Pathway permutation analysis using unstimulated cells: {wildcards.strictness} {wildcards.sites}" '
        '--to_working_dir ../../ '


rule run_de_pathway_analysis_unst_cells_cellcycle:
    input:
        sce=lambda wildcards: sce_list[wildcards.strictness][wildcards.sites]
    output:
        html='reports/de_pathway/de_pathway.unst_cells.cellcycle_analyses.{strictness}.{sites}.html',
        unst_rds='data/de_analysis_FTv62/cellcycle_analyses/{strictness}.{sites}.de_results_unstimulated_cells.cc.rds'
    conda:
        "envs/myenv.yaml"
    singularity:
        "docker://davismcc/r-singlecell-img"
    shell:
        'Rscript src/R/compile_report_de_pathways.R '
        '-c {wildcards.strictness}.{wildcards.sites} '
        '-o {output.html} '
        '--template src/Rmd/DE_pathways_FTv62_callset_clones_pairwise_vs_base.cell_cycle.unst_cells.Rmd '
        '--title "DE Pathway Analysis using unstimulated cells accounting for cell cycle : {wildcards.strictness} {wildcards.sites}" '
        '--to_working_dir ../../ '


rule run_de_pathway_analysis_unst_cells:
    input:
        sce=lambda wildcards: sce_list[wildcards.strictness][wildcards.sites]
    output:
        html='reports/de_pathway/de_pathway.unst_cells.{strictness}.{sites}.html',
        unst_rds='data/de_analysis_FTv62/{strictness}.{sites}.de_results_unstimulated_cells.rds'
    conda:
        "envs/myenv.yaml"
    singularity:
        "docker://davismcc/r-singlecell-img"
    shell:
        'Rscript src/R/compile_report_de_pathways.R '
        '-c {wildcards.strictness}.{wildcards.sites} '
        '-o {output.html} '
        '--template src/Rmd/DE_pathways_FTv62_callset_clones_pairwise_vs_base.unst_cells.Rmd '
        '--title "DE Pathway Analysis using unstimulated cells: {wildcards.strictness} {wildcards.sites}" '
        '--to_working_dir ../../ '


rule run_cell_assignment:
    input:
        can='data/canopy/canopy_results.{donor}.{strictness}.{sites}.rds',
        sce='data/sces/sce_{donor}_qc.rds',
        vcf='data/raw/mpileup/{donor}.mpileup.vcf.gz',
        csi='data/raw/mpileup/{donor}.mpileup.vcf.gz.csi'
    output:
        html = 'reports/cell_assignment/cell_assignment.{donor}.{strictness}.{sites}.html',
        sce = 'data/sces/sce_{donor}_with_clone_assignments.{strictness}.{sites}.rds',
        card = 'data/cell_assignment/cardelino_results.{donor}.{strictness}.{sites}.rds'
    conda:
        "envs/myenv.yaml"
    singularity:
        "docker://davismcc/r-singlecell-img"
    shell:
        'Rscript src/R/compile_report_cell_assign.R '
        '-i {input.sce} --vcf_file {input.vcf} --tree_file {input.can} '
        '-o {output.html} --results_sce {output.sce} --results_card {output.card} '
        '--template src/Rmd/cell_assignment_template.Rmd '
        '--title "Assigning single cells to clones: {wildcards.donor}" '
        '--donor {wildcards.donor} --to_working_dir ../../ '


rule run_canopy_donor_specific_coverage:
    input:
        'Data/exome-point-mutations/high-vs-low-exomes.v62.ft.{strictness}-{donor}.txt.gz'
    output:
        html = 'reports/canopy/canopy.analysis.{donor}.{strictness}.cell_coverage_sites.html',
        rds = 'data/canopy/canopy_results.{donor}.{strictness}.cell_coverage_sites.rds'
    conda:
        "envs/myenv.yaml"
    singularity:
        "docker://davismcc/r-singlecell-img"
    shell:
        'Rscript '
        'src/R/compile_report.R -i {input} -o {output.html} '
        '--results_out {output.rds} '
        '--template src/Rmd/canopy_analysis_template.Rmd '
        '--title "Canopy analysis: {wildcards.donor}" '
        '--donor {wildcards.donor} --to_working_dir ../../ '


rule run_canopy:
    input:
        'Data/exome-point-mutations/high-vs-low-exomes.v62.ft.{strictness}-alldonors.txt.gz'
    output:
        html = 'reports/canopy/canopy.analysis.{donor}.{strictness}.all_filt_sites.html',
        rds = 'data/canopy/canopy_results.{donor}.{strictness}.all_filt_sites.rds'
    conda:
        "envs/myenv.yaml"
    singularity:
        "docker://davismcc/r-singlecell-img"
    shell:
        'Rscript '
        'src/R/compile_report.R -i {input} -o {output.html} '
        '--results_out {output.rds} '
        '--template src/Rmd/canopy_analysis_template.Rmd '
        '--title "Canopy analysis: {wildcards.donor}" '
        '--donor {wildcards.donor} --to_working_dir ../../ '


rule filter_somatic_variants_per_donor_strict:
    input:
        flat='data/exome-point-mutations/high-vs-low-exomes.v62.ft.filt_strict-alldonors.txt.gz',
        vcf='data/raw/mpileup/{donor}.mpileup.vcf.gz',
        csi='data/raw/mpileup/{donor}.mpileup.vcf.gz.csi'
    output:
        'data/exome-point-mutations/high-vs-low-exomes.v62.ft.filt_strict-{donor}.txt.gz'
    conda:
        "envs/myenv.yaml"
    shell:
        'Rscript src/R/filter_variants.R -i {input.flat} -o {output} '
        '--donor_cell_vcf {input.vcf} --max_fdr 0.2 '
        '--min_prop_covered_cells 0.005 --donor_name {wildcards.donor}'


rule filter_somatic_variants_per_donor_lenient:
    input:
        flat='data/exome-point-mutations/high-vs-low-exomes.v62.ft.filt_lenient-alldonors.txt.gz',
        vcf='data/raw/mpileup/{donor}.mpileup.vcf.gz',
        csi='data/raw/mpileup/{donor}.mpileup.vcf.gz.csi'
    output:
        'data/exome-point-mutations/high-vs-low-exomes.v62.ft.filt_lenient-{donor}.txt.gz'
    conda:
        "envs/myenv.yaml"
    singularity:
        "docker://davismcc/r-singlecell-img"
    shell:
        'Rscript src/R/filter_variants.R -i {input.flat} -o {output} '
        '--donor_cell_vcf {input.vcf} --max_fdr 0.2 '
        '--min_prop_covered_cells 0.005 --donor_name {wildcards.donor}'


rule filter_somatic_variants_strict:
    input:
        'data/exome-point-mutations/high-vs-low-exomes.v62.ft.txt.gz'
    output:
        'data/exome-point-mutations/high-vs-low-exomes.v62.ft.filt_strict-alldonors.txt.gz'
    conda:
        "envs/myenv.yaml"
    singularity:
        "docker://davismcc/r-singlecell-img"
    shell:
        'Rscript src/R/filter_variants.R -i {input} -o {output} '
        '--max_fdr 0.05 --min_vaf_fibro 0.03 --max_vaf_fibro 0.45 '
        '--min_nalt_fibro 2.5 --max_vaf_ips 0.7 --combo_max_vaf_fibro 0.35 '
        '--combo_max_vaf_ips 0.3'


rule filter_somatic_variants_lenient:
    input:
        'data/exome-point-mutations/high-vs-low-exomes.v62.ft.txt.gz'
    output:
        'data/exome-point-mutations/high-vs-low-exomes.v62.ft.filt_lenient-alldonors.txt.gz'
    conda:
        "envs/myenv.yaml"
    singularity:
        "docker://davismcc/r-singlecell-img"
    shell:
        'Rscript src/R/filter_variants.R -i {input} -o {output} '
        '--max_fdr 0.1 --min_vaf_fibro 0.01 --max_vaf_fibro 0.45 '
        '--min_nalt_fibro 1.5 --max_vaf_ips 0.8 --combo_max_vaf_fibro 0.45 '
        '--combo_max_vaf_ips 0.45'

