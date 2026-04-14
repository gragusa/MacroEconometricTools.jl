# see documentation at https://juliadocs.github.io/Documenter.jl/stable/

using Documenter, MacroEconometricTools

makedocs(
    modules = [MacroEconometricTools],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Giuseppe Ragusa",
    sitename = "MacroEconometricTools.jl",
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "tutorials/getting_started.md",
            "tutorials/svar_iv.md",
            "tutorials/sign_restrictions.md",
            "tutorials/parallel_computing.md"
        ],
        "Mathematical Theory" => "mathematical/theory.md"
    ]
)

deploydocs(
    repo = "github.com/gragusa/MacroEconometricTools.jl.git",
    push_preview = true
)
