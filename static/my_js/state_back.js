var heatMap_svg = d3.select("#center_heatMap_svg");
var imageGrid_svg = d3.select("#imageGrid_svg");
var imageTypeGrid_svg = d3.select("#imageTypeGrid_svg");
var heatMap_svg_right = d3.select("#center_heatMap_svg_right");
var imageGrid_svg_right = d3.select("#imageGrid_svg_right");
var imageTypeGrid_svg_right = d3.select("#imageTypeGrid_svg_right");
var img_number = 0;

// 保存当前状态~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// const setDataSaveBtn = document.querySelector("#data_save")
const setDataSaveBtn = document.getElementById("data_name")

const selectElement = document.getElementById('state_name');
//
// setDataSaveBtn.addEventListener("click", function () {
//     console.log("xscale",xScale)
//      //让等待图表转起来
//     var spinner = d3.selectAll(".spinner-border");
//     spinner.style("display", "block");
//
//     const dataNameInput = document.getElementById('data_name');
//     const name = dataNameInput.value.trim(); // 获取输入框中的名称，并去除首尾空格
//     save_state(name);
//     spinner.style("display", "none");
//
// })

setDataSaveBtn.addEventListener('change', function() {
   console.log("xscale",xScale)
     //让等待图表转起来
    var spinner = d3.selectAll(".spinner-border");
    spinner.style("display", "block");

    const name = setDataSaveBtn.value.trim(); // 获取输入框中的名称，并去除首尾空格
    save_state(name);
    spinner.style("display", "none");

});
// 状态保存函数
async function save_state(name) {
    var dataset_type = document.getElementById("select_dataset_type_selection").value //在Document.querySelector()通过id获取才需要加#


    const state = {
        name:name,
        dataset_type: dataset_type,
        // xScale: xScale,
        // yScale: yScale,
        extent: extent,
        bins: bins,
        coordinates: coordinates,
        robustness_dict: robustness_dict,
        confidence_dict: confidence_dict,
        confidence_fore_dict: confidence_fore_dict,
        img_DNN_for_output_lst_400_dict: img_DNN_for_output_lst_400_dict,
        img_DNN_output_lst_400_dict: img_DNN_output_lst_400_dict,
        img_coords_lst_400_dict: img_coords_lst_400_dict,
        conf_matrix_dic: conf_matrix_dic,
        conf_matrix_label_x_dic: conf_matrix_label_x_dic,
        conf_matrix_label_y_dic: conf_matrix_label_y_dic,
        conf_matrix_acc_x_dic: conf_matrix_acc_x_dic,
        conf_matrix_acc_y_dic: conf_matrix_acc_y_dic,
        thresholds: thresholds,
        xAxis: xAxis,
        yAxis: yAxis,
        gX: gX,
        gY: gY,
        gX_right: gX_right,
        gY_right: gY_right,
        coordinates_offset: coordinates_offset
    };

    await axios.post("/save_state", {"state": state})
     .then(function (response) {
         data = response.data;
         selectElement.innerHTML = ''; // Clear existing options
         data.forEach(option => {
            const opt = document.createElement('option');
            opt.value = option.id;
            opt.textContent = option.name;
            selectElement.appendChild(opt);
         });

     })



}
// // 从某个保存好的状态开始
// const setDatauploadBtn = document.querySelector("#data_upload")
// setDatauploadBtn.addEventListener("click", function () {
//      //让等待图表转起来
//     var spinner = d3.selectAll(".spinner-border");
//     spinner.style("display", "block");
//
//     const dataNameInput = document.getElementById('state_name');
//     const name = dataNameInput.value; // 获取输入框中的名称，并去除首尾空格
//     upload_state(name);
// })


  // Add an event listener to capture the selected value
selectElement.addEventListener('change', function() {
     var spinner = d3.selectAll(".spinner-border");
    spinner.style("display", "block");
    const name = selectElement.options[selectElement.selectedIndex].text;

    upload_state(name);
    spinner.style("display", "none");


});
async function upload_state(name) {


    var dataset_type = document.getElementById("select_dataset_type_selection").value //在Document.querySelector()通过id获取才需要加#


    await axios.post("/upload_state", {"state_name": name,"dataset_type": dataset_type })
    .then(function (response) {
            data = response.data;
            console.log("xscale",data.xScale)

            extent = data.extent;
            bins = data.bins;
            xScale = d3.scaleLinear()
                .domain([extent.start_x, extent.end_x])
                .range([0, heatMap_width]);
            // 定义Y轴scale
            yScale = d3.scaleLinear()
                .domain([extent.start_y, extent.end_y])
                .range([0, heatMap_height]);
            coordinates = data.coordinates;
            confidence_dict = data.confidence_dict;
            confidence_fore_dict = data.confidence_fore_dict;
            img_DNN_for_output_lst_400_dict = data.img_DNN_for_output_lst_400_dict;
            img_DNN_output_lst_400_dict = data.img_DNN_output_lst_400_dict;
            img_coords_lst_400_dict = data.img_coords_lst_400_dict;
            conf_matrix_dic = data.conf_matrix_dic;
            conf_matrix_label_x_dic = data.conf_matrix_label_x_dic;
            conf_matrix_label_y_dic = data.conf_matrix_label_y_dic;
            conf_matrix_acc_y_dic = data.conf_matrix_acc_y_dic;
            conf_matrix_acc_x_dic = data.conf_matrix_acc_x_dic;
            thresholds = data.thresholds;
            xAxis = data.xAxis;
            yAxis = data.yAxis;
            gX = data.gX;
            gY = data.gY;
            gX_right = data.gX_right;
            gY_right = data.gY_right;
            coordinates_offset = data.coordinates_offset;
             //绘制热力图
            change_heatMap()
            //绘制网格图片概览
            change_imageGrid()
            //绘制类别概览
            change_imageTypeGrid()
            // 绘制混淆矩阵
            change_matrix()
            //默认是平移操作
            panBtn.click()
            //让等待图表隐藏起来
            spinner.style("display", "none");
        })
}


