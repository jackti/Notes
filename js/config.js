window.MathJax = {
    tex2jax: {
        inlineMath: [
            ["\\(", "\\)"]
        ],
        displayMath: [
            ["\\[", "\\]"]
        ]
    },
    TeX: {
        TagSide: "right",
        TagIndent: ".8em",
        MultLineWidth: "85%",
        useLabelIds: true,
        equationNumbers: {
            autoNumber: "AMS",
        },
        unicode: {
            fonts: "STIXGeneral,'Arial Unicode MS'"
        }
    },
    options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex"
    },
    displayAlign: "center",
    showProcessingMessages: false,
    messageStyle: "none"
};