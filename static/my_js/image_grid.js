/*
    这个js主要用来绘制右边的网格图片概览
*/

//修改右边的网格图片概览
function change_imageGrid() {
    var imageGrid_svg = d3.select("#imageGrid_svg");
    imageGrid_svg.selectAll("image").remove(); //先删除现有的
    var imageGrid_svg_right = d3.select("#imageGrid_svg_right");
    imageGrid_svg_right.selectAll("image").remove(); //先删除现有的
    draw_imageGrid();
}

//绘制右边图片概览
function draw_imageGrid() {
    // 停止转动加载图标
    var spinner = d3.selectAll(".spinner-border");
    spinner.style("display", "none");

    var svg_imageGrid = d3.select("#imageGrid_svg")
        .attr("width", imageGrid_width)
        .attr("height", imageGrid_height)
    var svg_imageGrid_right = d3.select("#imageGrid_svg_right")
        .attr("width", imageGrid_width)
        .attr("height", imageGrid_height)

    var rectWidth = 40

    var svg_imageGrid_g = svg_imageGrid.select("#imageGrid_g");
    var svg_imageGrid_g_right = svg_imageGrid_right.select("#imageGrid_g_right");
    // console.log("image_grid.js中img_coords_lst_400_dict: ", img_coords_lst_400_dict)
    // console.log("image_grid.js中robustness_dict: ", robustness_dict)


    var img_coords_lst_400 = img_coords_lst_400_dict["M1"]
    //添加边框
    svg_imageGrid_g.selectAll("rect")
        .data(img_coords_lst_400)
        .join("rect")
        .attr("class", "img_border_rect")
        .attr("fill", "none")
        .attr("stroke", "black")
        .attr("stroke-width", "1")
        .attr("height", rectWidth)
        .attr("width", rectWidth)
        .attr("x", (d, i) => rectWidth * (i % 20))
        .attr("y", (d, i) => rectWidth * Math.floor(i / 20));
    svg_imageGrid_g_right.selectAll("rect")
        .data(img_coords_lst_400)
        .join("rect")
        .attr("class", "img_border_rect")
        .attr("fill", "none")
        .attr("stroke", "black")
        .attr("stroke-width", "1")
        .attr("height", rectWidth)
        .attr("width", rectWidth)
        .attr("x", (d, i) => rectWidth * (i % 20))
        .attr("y", (d, i) => rectWidth * Math.floor(i / 20));

    var dataset_type = document.getElementById("select_dataset_type_selection").value
    //添加图片
    svg_imageGrid_g.selectAll("image")
        .data(img_coords_lst_400)
        .join("image")
        .attr("number", (d, i) => i) //方便后续定位图片
        .attr("href", (d, i) => `../static/data/${dataset_type}/pic/grid_images/grid_image_${i}.png?t=` + Math.random())
        .attr("height", rectWidth)
        .attr("width", rectWidth)
        .attr("x", (d, i) => rectWidth * (i % 20))
        .attr("y", (d, i) => rectWidth * Math.floor(i / 20))
        .attr("data-bs-toggle", "tooltip")
        .attr("title", "image")
        .attr("border", "5")
     svg_imageGrid_g_right.selectAll("image")
        .data(img_coords_lst_400)
        .join("image")
        .attr("number", (d, i) => i) //方便后续定位图片
        .attr("href", (d, i) => `../static/data/${dataset_type}/pic/grid_fore_images/grid_fore_image_${i}.png?t=` + Math.random())
        .attr("height", rectWidth)
        .attr("width", rectWidth)
        .attr("x", (d, i) => rectWidth * (i % 20))
        .attr("y", (d, i) => rectWidth * Math.floor(i / 20))
        .attr("data-bs-toggle", "tooltip")
        .attr("title", "image")
        .attr("border", "5")
    
}
