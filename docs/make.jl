push!(LOAD_PATH,"./src/")
using Documenter, FeedbackParticleFilters

makedocs(
    modules = [FeedbackParticleFilters],
    format = Documenter.HTML(),
    clean = false,              # do we clean build dir
    sitename = "FeedbackParticle Filters.jl",
    authors = "Simone Carlo Surace",
    pages = Any[ # Compat: `Any` for 0.4 compat
        "Home" => "index.md",
        "Manual" => Any[
             "getting_started.md",
             "filtering.md",
             "fpf.md",
             "hidden.md",
             "observation.md",
             "gainest.md",
             "reference.md",
        ],
    ],
)

#deploydocs(
   # repo = "github.com/simsurace/FeedbackParticleFilters.jl.git",
  #  target = "build",
    # julia = "1.0",
    # osname = "linux",
    # make = nothing,
    # deps = nothing,
    # deps   = Deps.pip("mkdocs", "python-markdown-math"),
#)
