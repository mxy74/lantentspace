/*
    这个js的功能主要是绘制类别网格
*/
//图片类别颜色
// 师兄原版颜色
// var labels_colors = { 0: '#8dd3c7', 1: '#ffffb3', 2: '#bebada', 3: '#fb8072', 4: '#80b1d3', 5: '#fdb462', 6: '#b3de69', 7: '#fccde5', 8: '#d9d9d9', 9: '#bc80bd', 10: "white"};
// colorbrewer颜色
var labels_colors = { 0: '#a6cee3', 1: '#1f78b4', 2: '#b2df8a', 3: '#33a02c', 4: '#fb9a99', 5: '#e31a1c', 6: '#fdbf6f', 7: '#ff7f00', 8: '#cab2d6', 9: '#6a3d9a', 10: "white" };
// var steeringAngle_labels_colors = { 0: '#a6cee3', 1: '#1f78b4', 2: '#b2df8a', 3: '#33a02c', 4: '#fb9a99', 5: '#e31a1c', 6: '#fdbf6f', 7: '#ff7f00', 8: '#cab2d6', 9: '#6a3d9a', 10: "white"};

//修改右边的网格类别概览
function change_matrix() {
    var confusion_matrix_g = d3.select("#confusion_matrix_g");
    confusion_matrix_g.selectAll("rect").remove(); //先删除现有的
    confusion_matrix_g.selectAll("text").remove(); //先删除现有的
    draw_m();
}
function draw_m(){

    var data = conf_matrix_dic["M1"]
    // const labels = ["0", "1", "2", "3", "4","5", "6", "7", "8","9"];
    const labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];

    const margin = { top: 10, right: 10, bottom: 45, left: 45 };
    const width = 380 - margin.left - margin.right;
    const height = 360 - margin.top - margin.bottom;

    const svg = d3.select("#confusion_matrix_svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("id", "confusion_matrix_g")
        .attr("transform", `translate(${margin.left}, ${margin.top})`);

    const x = d3.scaleBand()
        .domain(d3.range(data.length))
        .range([0, width]);

    const y = d3.scaleBand()
        .domain(d3.range(data.length))
        .range([0, height]);


    // const colors = [
    //     "#edf8b1",
    //     "#c7e9b4","#7fcdbb",
    //     "#41b6c4"
    //     // , "#1d91c0"
    // ];
    const colors = [
    '#fee5d9', '#fdd7c5', '#fcc9b1', '#fbba9d','#faac89', '#faa57f'
        ]
        // "#225ea8","#253494","#081d58"];
    const colorScale = d3.scaleQuantile()
        .domain([0, d3.max(data.flat())])
        .range(colors);

    // Draw cells
    svg.selectAll("#confusion_matrix_g")
        .data(data.flat())
        .join("rect")
        .attr("class", "cell")
        .attr("x", (d, i) => x(i % data.length))
        .attr("y", (d, i) => y(Math.floor(i / data.length)))
        .attr("width", x.bandwidth())
        .attr("height", y.bandwidth())
        .attr("fill", d => colorScale(d))
        .attr("stroke", '#A1A1A1')
        .attr("stroke-width", '2px')
        .attr("rx", 5) // 设置 x 轴圆角半径
        .attr("ry", 5); // 设置 y 轴圆角半径

    // Add labels to cells
    svg.selectAll("confusion_matrix_g")
        .data(data.flat())
        .join("text")
        .attr("class", "cell-label")
        .attr("x", (d, i) => x(i % data.length) + x.bandwidth() / 2)
        .attr("y", (d, i) => y(Math.floor(i / data.length)) + y.bandwidth() / 2)
        .attr("dy", ".35em")
        .text(d => d)
        .attr("font-size", "12px")
        .attr("font-family", "Consolas, courier")
        .style("text-anchor", "middle")
        .style("fill", "#726f6f");



    // Add labels for rows and columns
    svg.selectAll("confusion_matrix_g")
        .data(labels)
        .join("text")
        .attr("class", "label")
        .attr("x", -8)
        .attr("y", (d, i) => y(i) + y.bandwidth() / 2)
        .attr("dy", ".35em")
        .text(d => d)
        .style("text-anchor", "end")
        .attr("font-size", "12px")
        .attr("font-family", "Consolas, courier")
        .style("fill", "#726f6f")
        .attr("transform", (d, i) => {
            const xPos = -5;
            const yPos = y(i) + y.bandwidth() / 2;
            return `rotate(-30, ${xPos}, ${yPos})`;
        });

    svg.selectAll("confusion_matrix_g")
        .data(labels)
        .join("text")
        .attr("class", "label")
        .attr("x", (d, i) => x(i) + x.bandwidth() / 2)
        .attr("y", height + 12)
        .attr("dy", ".35em")
        .text(d => d)
        .style("text-anchor", "middle")
        .attr("font-size", "12px")
        .attr("font-family", "Consolas, courier")
        .style("fill", "#726f6f")
        .attr("transform", (d, i) => {
            const xPos = x(i) + x.bandwidth() / 2;
            const yPos = height + 13;
            return `rotate(-30, ${xPos}, ${yPos})`;
    });

}


