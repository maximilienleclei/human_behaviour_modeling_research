You have a tendency to write my name as maximilian instead of maximilien in certain commands, which messes up your file reading, beware!

In plan mode, make sure to dump your plan to a file in the repository in case we run out of context/usage. Delete it once the plan has been implemented. Also write down progress checks so that we know what you already did and what is left to do.

Do not ever worry about backwards compatibility, I would always rather simplify and clean things up to build better than stay attached to older code.

Every `.py` file over 30 lines must have an equivalent `.md` file with a high-level explanation of what goes on in that file: first what it is for, then what it contains. After every new file/file edit make absolutely sure that you update the corresponding `.md` file accordingly. Then, at the root of every directory we must have a `__init__.md` file that is a high-level explanation of all the files (and nested files) in that directory. Make sure to take a look at these too when editing files in a directory. If a lot of changes have been made it is possible that the parent directory's `__init__.md` has to be updated too so be on the lookout for that.
Also make sure that you utilize that functionality. There are times where reading the `.md` files are sufficient and you do not need to load their corresponding `.py` files into context.

When starting a new session, always read `@README.md`, it indexes other useful `.md` files in `docs/`. Make sure you read these if they look pertinent, and also make sure to keep them up to date when you make modifications to the codebase.

--time=00:30:00 --gpus=h100_1g.10gb:1 --account=rrg-pbellec --mem=15G --cpus-per-task=2