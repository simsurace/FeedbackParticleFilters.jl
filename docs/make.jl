push!(LOAD_PATH,"/home/simone/Documents/uni/postdoc/projects/FeedbackParticleFilters.jl/")
using Documenter, FeedbackParticleFilters

makedocs(
    modules = [FeedbackParticleFilters],
    format = Documenter.HTML(),
    clean = false,              # do we clean build dir
    sitename = "FeedbackParticle Filters.jl",
    authors = "Simone Carlo Surace",
    pages = Any[ # Compat: `Any` for 0.4 compat
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Background" => Any[
             "filtering.md",
             "fpf.md",
        ],
        "Manual" => Any[
             "hidden.md",
             "observation.md",
             "gainest.md",
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
deploydocs(
    repo = "github.com/simsurace/FeedbackParticleFilters.jl.git",
)