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

    var img_fore_labels_lst_400 = img_DNN_for_output_lst_400_dict["M1"]
    console.log(img_fore_labels_lst_400)
    labels_colors = { 0: '#a6cee3', 1: '#1f78b4', 2: '#b2df8a',
        3: '#33a02c', 4: '#fb9a99', 5: '#e31a1c', 6: '#fdbf6f',
        7: '#ff7f00', 8: '#cab2d6', 9: '#6a3d9a' };
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
        .attr("fill", (d, i) => labels_colors[parseInt(img_fore_labels_lst_400[i])])
        .attr("stroke", "white")
        .attr("stroke-width", "0.5")
        // .attr("stroke", (d, i) => labels_colors[img_fore_labels_lst_400[i]]) // 根据类别信息设置边框颜色
        // .attr("stroke-width", "3") // 设置边框宽度为2像素
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
        .attr("height", rectWidth-6)
        .attr("width", rectWidth-6)
        .attr("x", (d, i) => rectWidth * (i % 20)+3)
        .attr("y", (d, i) => rectWidth * Math.floor(i / 20)+3)
        .attr("data-bs-toggle", "tooltip")
        .attr("title", "image")
        .attr("border", "5")
            // 添加CAM图像层

    const camImages = svg_imageGrid_g_right.selectAll(".cam-image")
        .data(img_coords_lst_400)
        .join("image")
        // .attr("class", "cam-image")
        .attr("number", (d, i) => i) // 方便后续定位图片
        .attr("href", (d, i) => `../static/data/${dataset_type}/pic/cam_image/cam_image_${i}.png?t=` + Math.random())
        .attr("height", rectWidth - 6)
        .attr("width", rectWidth - 6)
        .attr("x", (d, i) => rectWidth * (i % 20) + 3)
        .attr("y", (d, i) => rectWidth * Math.floor(i / 20) + 3)
        .attr("data-bs-toggle", "tooltip")
        .attr("title", "CAM image")
        // .attr("opacity", 0.5); // 设置透明度以便原始图像可见
    const camBtn = document.querySelector("#radio_cam");
    camBtn.addEventListener("click", function () {
        const currentOpacity = camImages.attr("opacity");
        const newOpacity = currentOpacity === "0" ? "1" : "0";
        camImages.attr("opacity", newOpacity);
     })
}
