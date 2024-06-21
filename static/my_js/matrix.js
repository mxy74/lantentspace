
// const GTSRB_classes = [
//   '20_speed',
//   '30_speed',
//   '50_speed',
//   '60_speed',
//   '70_speed',
//   '80_speed',
//   '80_lifted',
//   '100_speed',
//   '120_speed',
//   'no_overtaking_general',
//   'no_overtaking_trucks',
//   'right_of_way_crossing',
//   'right_of_way_general',
//   'give_way',
//   'stop',
//   'no_way_general',
//   'no_way_trucks',
//   'no_way_one_way',
//   'attention_general',
//   'attention_left_turn',
//   'attention_right_turn',
//   'attention_curvy',
//   'attention_bumpers',
//   'attention_slippery',
//   'attention_bottleneck',
//   'attention_construction',
//   'attention_traffic_light',
//   'attention_pedestrian',
//   'attention_children',
//   'attention_bikes',
//   'attention_snowflake',
//   'attention_deer',
//   'lifted_general',
//   'turn_right',
//   'turn_left',
//   'turn_straight',
//   'turn_straight_right',
//   'turn_straight_left',
//   'turn_right_down',
//   'turn_left_down',
//   'turn_circle',
//   'lifted_no_overtaking_general',
//   'lifted_no_overtaking_trucks'
// ];
//修改右边的网格类别概览
function change_matrix() {
    var confusion_matrix_g = d3.select("#confusion_matrix_g");
    confusion_matrix_g.remove();
    // confusion_matrix_g.selectAll("rect").remove(); //先删除现有的
    // confusion_matrix_g.selectAll("text").remove(); //先删除现有的
    draw_m();
}
function draw_m(){

    var data = conf_matrix_dic["M1"]
    console.log(conf_matrix_label_dic)
    var label = conf_matrix_label_dic["M1"]
    // const labels = ["0", "1", "2", "3", "4","5", "6", "7", "8","9"];
     var dataset_type = document.getElementById("select_dataset_type_selection").value //在Document.querySelector()通过id获取才需要加#
    console.log("dataset_type: ", dataset_type)
    let labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
    if (dataset_type === "GTSRB") {
        labels = label.map(index => GTSRB_classes[index]);
    }

    const margin = { top: 10, right: 10, bottom: 45, left: 45 };
    const width = data.length*30 - margin.left - margin.right;
    const height = data.length*30 - margin.top - margin.bottom;

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
    '#fee5d9', '#fdd7c5', '#fcc9b1', '#fbba9d',
        '#faac89', '#faa57f'
        ]
    // 找除了对角线以外最大的数字
    const flattenedData = data.flat();
    const offDiagonalData = flattenedData.filter((d, i) => Math.floor(i / data.length) !== (i % data.length));

    const colorScale = d3.scaleQuantile()
    .domain([0, d3.max(offDiagonalData)])
    .range(colors);
        // "#225ea8","#253494","#081d58"];
    // const colorScale = d3.scaleQuantile()
    //     .domain([0, d3.max(data.flat())])
    //     .range(colors);

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
        .attr("fill", (d, i) => {
        if (Math.floor(i / data.length) === (i % data.length)) {
            return "#FFFFFF"; // 对角线位置颜色设置为白色或你希望的颜色
        }
        return colorScale(d);
    })
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


    if(dataset_type === "CIFAR10"){

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
        });}
    if(dataset_type === "GTSRB") {
        // 添加行图标
        svg.selectAll("confusion_matrix_g")
            .data(label)
            .join("image")
            .attr("class", "label-icon")
            .attr("xlink:href", d => `../static/example/label_pic/${d}.png`)  // 图标文件路径
            .attr("width", 20)
            .attr("height", 20)
            .attr("x", -30)
            .attr("y", (d, i) => y(i) + y.bandwidth() / 2 - 10);  // 中心对齐图标

        // 添加列图标
        svg.selectAll("confusion_matrix_g")
            .data(label)
            .join("image")
            .attr("class", "label-icon")
            .attr("xlink:href", d => `../static/example/label_pic/${d}.png`)  // 图标文件路径
            .attr("width", 20)
            .attr("height", 20)
            .attr("x", (d, i) => x(i) + x.bandwidth() / 2 - 10)  // 中心对齐图标
            .attr("y", height + 5);

    }

}


