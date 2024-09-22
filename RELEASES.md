# 2024-09-22

- Early stop when loss becomes NaN
- Fix a bug in load_best_model

# 2024-09-20

- Add `utils.select_project`

# 2024-09-10

- Support `TPE.GridSampler`
  - Add `OptimizeConfig.grid_search_space`
  - Modify `OptimizeConfig._create_sampler`
  - **Caution**: For GridSampler, all variables in search_space should be categorical
