refactor WIP

Legacy Standalone Optimizers (RHC, SA, GA)
If you are upgrading from Pyperch ≤ 0.1.6, the original “standalone” optimizers
(rhc, sa, ga functions with manual loops) have been moved but are still accessible.

You can find the previous implementations here:

Git tag: TAG

Directory: PATH (archived copy from pre-refactor branch)

These remain available for users who prefer the original functional API or need to debug compatibility issues with older projects. The new refactored versions are available in pyperch.optim.* and integrated with the unified Trainer API.
