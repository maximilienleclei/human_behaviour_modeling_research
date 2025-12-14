CRITICAL PATH REMINDER:
- The repository path contains "maximilienleclei" (NOT "maximilianleclei" or "maximilian")
- ALWAYS use Glob first to get the correct full path
- ALWAYS copy paths exactly from Glob/Bash output - NEVER type them manually
- If you see a path error, check character-by-character that you copied it exactly

In plan mode, make sure to dump your plan to a file in the repository in case we run out of context/usage. Delete it once the plan has been implemented. Also write down progress checks so that we know what you already did and what is left to do.

Do not ever worry about backwards compatibility, I would always rather simplify and clean things up to build better than stay attached to older code.

Every `.py` file over 30 lines must have an equivalent `.md` file with a high-level explanation of what goes on in that file: first what it is for, then what it contains. It should be around 1/10th of the character count of the `.py` file. Then, at the root of every directory we must have a `__init__.md` file that is a high-level explanation of all the files (and nested files) in that directory. Make sure to take a look at these too when editing files in a directory. 

MANDATORY WORKFLOW when editing .py files:
  1. Edit the .py file
  2. IMMEDIATELY edit the corresponding .md file in THE SAME response
  3. If a lot of changes have been made it is possible the folder's `__init__.md` has to be updated too (and its parents', grandparents', etc) so be on the lookout for that.
  3. NEVER proceed to other tasks until BOTH files are updated
  4. This is NON-NEGOTIABLE and BLOCKING

Also make sure that you utilize that functionality. There are times where reading the `.md` files are sufficient and you do not need to load their corresponding `.py` files into context.
