# Documentation Index

**Last Updated:** December 11, 2024

Welcome to the human behaviour modeling research documentation! This directory contains comprehensive documentation organized by purpose.

---

## Quick Start: Which Document Do I Read?

### 🎯 I'm New to This Project
**Start here:** [Root README](../README.md) → [ARCHITECTURE.md](ARCHITECTURE.md)

**Time:** 10-15 minutes

**You'll learn:**
- What this project does
- How the system is designed
- Key concepts and terminology

---

### 🏃 I Want to Run Experiments
**Go to:** [USAGE_GUIDE.md](USAGE_GUIDE.md)

**Time:** 5 minutes (for quick start), 30 minutes (for comprehensive reference)

**You'll learn:**
- All command-line options
- Common workflows
- Troubleshooting tips

---

### 🔍 I Need to Find Specific Code
**Go to:** [NAVIGATION.md](NAVIGATION.md)

**Time:** 1-2 minutes (quick lookup)

**You'll learn:**
- Where specific files are
- "Where do I find X?" lookups
- Import patterns

---

### 📊 I Want to Know Current Status
**Go to:** [STATUS.md](STATUS.md)

**Time:** 5 minutes

**You'll learn:**
- What's implemented
- What's in progress
- Known issues and gaps
- Next steps

---

### 🤔 I'm Starting a New Session
**Workflow:**
1. Read [STATUS.md](STATUS.md) (5 min) - What's done, what's next
2. Skim [NAVIGATION.md](NAVIGATION.md) (2 min) - Refresh on file locations
3. Reference [USAGE_GUIDE.md](USAGE_GUIDE.md) as needed - Copy-paste commands

**Total:** 7-10 minutes to get oriented

---

## Documentation Overview

### 📄 [../README.md](../README.md) (Root)
**Purpose:** Project entry point and overview

**Length:** 2-3 pages

**Contents:**
- One-paragraph project description
- Research question and approach
- Quick start commands
- Repository structure
- Links to all documentation
- Prerequisites and installation

**When to read:**
- First time exploring the project
- Showing project to someone new
- Need quick overview

**Update frequency:** Rarely (stable)

---

### 🏗️ [ARCHITECTURE.md](ARCHITECTURE.md)
**Purpose:** Understand how the system is designed

**Length:** 5-7 pages

**Contents:**
- Research context and goals
- High-level system design
- Module-by-module breakdown
  - platform/models/
  - platform/optimizers/
  - platform/data/
  - platform/evaluation/
  - experiments/tracking/
- Key design principles
- Technology stack
- Supported configurations matrix
- Data flow diagrams

**When to read:**
- Understanding system design
- Adding new features
- Debugging complex issues
- Architectural decisions

**Update frequency:** Rarely (stable unless major refactoring)

---

### 📚 [USAGE_GUIDE.md](USAGE_GUIDE.md)
**Purpose:** Complete reference for running experiments

**Length:** 6-8 pages

**Contents:**
- Quick start examples
- Platform runner documentation
  - All command-line arguments
  - Dataset options
  - Model options
  - Optimizer options
- Common workflows
  - Smoke test
  - Development test
  - Full training run
  - Parameter sweeps
- Cluster usage (SLURM)
- Monitoring and results
- Troubleshooting guide

**When to read:**
- Running your first experiment
- Learning all available options
- Debugging errors
- Need to run on cluster

**Update frequency:** Occasionally (when new features added)

---

### 🗺️ [NAVIGATION.md](NAVIGATION.md)
**Purpose:** Quick file lookup and navigation

**Length:** 3-4 pages

**Contents:**
- Complete directory tree with annotations
- Key files cheat sheet
- "Where to find X?" lookup table
- Import patterns
- File complexity guide

**When to read:**
- Looking for specific code
- Need to import a module
- Understanding file organization
- Quick reference during coding

**Update frequency:** Occasionally (when files added/moved)

---

### 📊 [STATUS.md](STATUS.md)
**Purpose:** Current work, what's done, what's next

**Length:** 4-6 pages

**Contents:**
- Current implementation status
  - Completed phases
  - Pending phases
- Recent work summary
  - What was added recently
  - Files modified
- Known issues and gaps
  - Critical issues
  - Non-critical gaps
  - Workarounds
- Immediate next steps
  - Testing priorities
  - Decision points
- Long-term roadmap

**When to read:**
- Starting a new session
- Planning next work
- Understanding current state
- Checking known issues

**Update frequency:** **FREQUENTLY** (after each major session or milestone)

**Note:** This is the ONLY document that changes frequently. All others are relatively stable.

---

### 🧩 [../platform/README.md](../platform/README.md)
**Purpose:** Platform-specific documentation

**Length:** 3-4 pages

**Contents:**
- Platform directory structure
- Quick start examples
- Available options summary
- Model architectures
- Optimizer algorithms
- Data loading
- Configuration
- Integration with tracking
- Design principles

**When to read:**
- Deep dive into platform internals
- Working specifically on platform code
- Understanding platform modules

**Update frequency:** Occasionally (when platform updated)

---

## Documentation Organization Principles

### 1. Separation of Concerns

Each document has a **single, clear purpose**:
- README → Project overview
- ARCHITECTURE → System design
- USAGE_GUIDE → How to run things
- NAVIGATION → Where to find things
- STATUS → Current work and progress

No overlap or redundancy.

### 2. Stable vs. Dynamic

**Stable** (rarely change):
- README.md (project overview)
- ARCHITECTURE.md (system design)
- USAGE_GUIDE.md (command reference)
- NAVIGATION.md (file locations)

**Dynamic** (changes frequently):
- STATUS.md (current work, updated after each session)

This separation means you don't have to re-read everything each session.

### 3. Scannable Format

All documents use:
- Clear section headings
- Tables for reference information
- Code blocks for examples
- Bullet points for lists
- "When to read" and "Update frequency" metadata

Easy to skim and find what you need.

### 4. Consistent Structure

Every document has:
- **Last Updated** date
- **Table of Contents**
- **Summary** or **Quick Reference** section
- **Related Documentation** links at the end

### 5. Multiple Entry Points

Different users have different needs:
- **Researchers** → STATUS.md → USAGE_GUIDE.md
- **Developers** → ARCHITECTURE.md → NAVIGATION.md
- **New users** → README.md → ARCHITECTURE.md
- **Quick reference** → NAVIGATION.md

---

## Documentation Maintenance

### When to Update Each Document

**After adding a new feature:**
1. Update [STATUS.md](STATUS.md) - Add to "Recent Work"
2. Update [USAGE_GUIDE.md](USAGE_GUIDE.md) - Document new commands/options
3. Update [NAVIGATION.md](NAVIGATION.md) - Add file locations (if new files)
4. Update [ARCHITECTURE.md](ARCHITECTURE.md) - Only if major architectural change

**After each work session:**
1. Update [STATUS.md](STATUS.md) - Recent work, next steps, known issues

**When project structure changes:**
1. Update [NAVIGATION.md](NAVIGATION.md) - Directory tree, file locations
2. Update [README.md](../README.md) - Repository structure (if major)

**Rarely needed:**
- [ARCHITECTURE.md](ARCHITECTURE.md) - Only for major refactoring
- [../README.md](../README.md) - Only for project scope changes

---

## Quick Reference: Document Characteristics

| Document | Pages | Update Freq | Primary Use Case |
|----------|-------|-------------|------------------|
| **README.md** | 2-3 | Rarely | Project overview, first read |
| **ARCHITECTURE.md** | 5-7 | Rarely | System design, understanding structure |
| **USAGE_GUIDE.md** | 6-8 | Occasionally | Running experiments, command reference |
| **NAVIGATION.md** | 3-4 | Occasionally | Finding files, quick lookup |
| **STATUS.md** | 4-6 | **Frequently** | Current work, session planning |
| **platform/README.md** | 3-4 | Occasionally | Platform internals |

---

## Documentation Reading Paths

### Path 1: New User Onboarding (30 minutes)
1. [README.md](../README.md) (5 min)
2. [ARCHITECTURE.md](ARCHITECTURE.md) (15 min)
3. [USAGE_GUIDE.md](USAGE_GUIDE.md) - Quick Start section (5 min)
4. [STATUS.md](STATUS.md) (5 min)

**Result:** Understand project, can run first experiment

---

### Path 2: Quick Session Start (5-10 minutes)
1. [STATUS.md](STATUS.md) (5 min) - What's done, what's next
2. [NAVIGATION.md](NAVIGATION.md) (2 min) - Refresh file locations
3. [USAGE_GUIDE.md](USAGE_GUIDE.md) - Reference as needed

**Result:** Oriented and ready to work

---

### Path 3: Debugging Session (variable)
1. [USAGE_GUIDE.md](USAGE_GUIDE.md) - Troubleshooting section (5-10 min)
2. [NAVIGATION.md](NAVIGATION.md) - Find relevant files (2 min)
3. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand component interaction (5-15 min)
4. [STATUS.md](STATUS.md) - Check known issues (2 min)

**Result:** Understand issue, know where to fix

---

### Path 4: Adding New Feature (variable)
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand design (10 min)
2. [NAVIGATION.md](NAVIGATION.md) - Find similar code (5 min)
3. [STATUS.md](STATUS.md) - Check roadmap and priorities (5 min)

**Result:** Know where and how to add feature

---

## Tips for Effective Documentation Use

### 1. Bookmark Key Documents
- Keep [NAVIGATION.md](NAVIGATION.md) open in a tab for quick reference
- Refer to [STATUS.md](STATUS.md) at start and end of each session

### 2. Use Search
- Ctrl+F / Cmd+F to find specific terms
- All docs designed to be searchable

### 3. Follow the Links
- Documents link to each other for related information
- "Related Documentation" section at end of each doc

### 4. Update STATUS.md Regularly
- Keep [STATUS.md](STATUS.md) current after each session
- Future you will thank present you

### 5. Don't Re-read Everything
- Only STATUS.md needs frequent re-reading
- Other docs are stable references

---

## Contributing to Documentation

### Documentation Standards

**When adding new documentation:**
1. Add entry to this index (docs/README.md)
2. Include metadata (Last Updated, purpose, update frequency)
3. Add table of contents for docs >2 pages
4. Use consistent formatting (headings, code blocks, tables)
5. Add "Related Documentation" links at end

**When updating existing documentation:**
1. Update "Last Updated" date
2. Maintain consistent formatting
3. Update related cross-references if needed

---

## Documentation Hierarchy

```
Root README.md                   (Entry point)
    ↓
    ├─→ docs/ARCHITECTURE.md     (System design - READ SECOND)
    ├─→ docs/USAGE_GUIDE.md      (How to run - READ FOR COMMANDS)
    ├─→ docs/NAVIGATION.md       (Where is code - QUICK REFERENCE)
    └─→ docs/STATUS.md           (Current work - READ EACH SESSION)
            ↓
        platform/README.md       (Platform internals - OPTIONAL)
```

---

## External References

### Official Documentation
- **PyTorch:** https://pytorch.org/docs/
- **HuggingFace Datasets:** https://huggingface.co/docs/datasets/
- **Gymnasium:** https://gymnasium.farama.org/
- **jaxtyping:** https://docs.kidger.site/jaxtyping/

### Research Papers
[Add relevant papers here as needed]

---

## Summary

This documentation system is designed to:
1. **Minimize orientation time** - 5-10 minutes to get started in new sessions
2. **Separate concerns** - Each doc has a clear, distinct purpose
3. **Reduce redundancy** - Information lives in one place
4. **Support different workflows** - Multiple entry points for different needs
5. **Stay maintainable** - Only STATUS.md changes frequently

**Quick Navigation:**
- Need overview? → [README.md](../README.md)
- Need to understand design? → [ARCHITECTURE.md](ARCHITECTURE.md)
- Need to run experiment? → [USAGE_GUIDE.md](USAGE_GUIDE.md)
- Need to find file? → [NAVIGATION.md](NAVIGATION.md)
- Need current status? → [STATUS.md](STATUS.md)

**Happy coding!** 🚀

---

**Meta:** This index itself should be updated when:
- New documentation files are added
- Documentation organization changes
- Reading paths need revision
