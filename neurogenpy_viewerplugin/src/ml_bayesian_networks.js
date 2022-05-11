$(document).ready(function () {
    var csrf_token = $("#csrf_token").val();
    var sigmajs_bn = {};
    var sigmajs_bn_settings = {
        "instance_id": "bayesian-network-graph",
        "drawing_engine": "webgl",
        "display_original_height": 600,
        "active_layout": "circular",
        "active_group": "",
        "active_category": "",
        "active_structure": "",
        "show_arrows": true,
        "original_color": "",
        "num_nodes": "",
        "num_edges": "",
        "color_common_edges": "",
        "color_structure_1": "",
        "color_structure_2": "",
        "highlight_node_color": "",
        "evidence_set_color": "#CC0000",
        "width_slider": $("#bn-graph-width-slider")[0],
        "height_slider": $("#bn-graph-height-slider")[0],
        "nodes_slider": $("#bn-nodes-size-slider")[0],
        "edges_slider": $("#bn-edges-size-slider")[0],
        "filter_edges_weight_slider": $("#slider-filter-edges-by-weight")[0],
        "structures_info": {}
    };

    $(".load-bn-dataset-example").click(function (e) {
        var waiting_div_elem = $("#waiting-continue-upload-dataset-bn");
        var error_elem = $("#error-continue-upload-dataset-bn");
        var success_elem = $("#success-continue-upload-dataset-bn");
        var next_section_elem = $("#section-learn-structure-bn");

        var data_client_json = {
            "dataset_type": $(this).attr("dataset-type"),
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json);
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_load_dataset_example/",
            data: data_send,
            dataType: 'json',
            beforeSend: function () {
                waiting_div_elem.removeClass("hide-element");
                waiting_div_elem.show();
            },
            success: function (data) {
                waiting_div_elem.hide();
                if (data["error"]) {
                    error_elem.text(data["error_message"]);
                    success_elem.hide();
                } else {
                    error_elem.empty();
                    success_elem.removeClass("hide-element");
                    success_elem.show();
                    next_section_elem.removeClass("hide-element");
                    next_section_elem.show();

                    $('html, body').animate({
                        scrollTop: next_section_elem.offset().top
                    }, 1000);
                }
            },
            error: function () {
                waiting_div_elem.hide();
                success_elem.show();
                error_elem.text("Server error");
            }
        });
    });

    $("#load-bn-discrete-example").click(function (e) {
        var waiting_div = $("#waiting-load-bn-example");
        var error_div = $("#error-load-bn-example");

        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
        };
        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_load_discrete_example/",
            data: data_send,
            dataType: 'json',
            beforeSend: function () {
                waiting_div.removeClass("hide-element").show();
            },
            success: function (data) {
                waiting_div.hide();
                if (data["error"]) {
                    error_div.text(data["error_message"]);
                    error_div.removeClass("hide-element").show();
                } else {
                    $("#continue-button-upload-bn").trigger("click");
                }
            },
            error: function (e) {
                waiting_div.hide();
                error_div.text(e);
                error_div.removeClass("hide-element").show();
            }
        });
    });

    $("#load-bn-continuous-example").click(function (e) {
        var waiting_div = $("#waiting-load-bn-example");
        var error_div = $("#error-load-bn-example");

        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
        };
        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_load_continuous_example/",
            data: data_send,
            dataType: 'json',
            beforeSend: function () {
                waiting_div.removeClass("hide-element").show();
            },
            success: function (data) {
                waiting_div.hide();
                if (data["error"]) {
                    error_div.text(data["error_message"]);
                    error_div.removeClass("hide-element").show();
                } else {
                    $("#continue-button-upload-bn").trigger("click");
                }
            },
            error: function (e) {
                waiting_div.hide();
                error_div.text(e);
                error_div.removeClass("hide-element").show();
            }
        });
    });

    $("#upload-bn-structure-1").fileinput({
        hideThumbnailContent: true, // hide image, pdf, text or other content in the thumbnail preview
        theme: "explorer",
        uploadUrl: "/morpho/ml_bn_load_only_structure/",
        fileActionSettings: {
            showDownload: false,
            showUpload: false,
            indicatorNew: '',
            removeIcon: '<i class="glyphicon glyphicon-trash fa-2x" aria-hidden="true"></i>',
            zoomIcon: '<i class="glyphicon glyphicon-zoom-in fa-2x"></i>',
        },
        previewClass: "file-input-small container-center",
        layoutTemplates: {
            main1: "<div class=\'container-center\'>\n" +
                "{preview}\n" +
                "<div class=\'input-group {class} file-input-browse-small\'>\n" +
                "   <div class=\'input-group-btn\ input-group-prepend file-input-button-small'>\n" +
                "       {browse}\n" +
                "       {upload}\n" +
                "       {remove}\n" +
                "   </div>\n" +
                "   {caption}\n" +
                "</div>" +
                "</div>"
        },
        browseIcon: "<i class=\"glyphicon glyphicon-folder-open icon-margin-right\"></i> ",
        uploadExtraData: function (previewId, index) {
            var obj = {};
            obj['csrfmiddlewaretoken'] = csrf_token;
            obj['structure'] = 1;
            return obj;
        },
        uploadAsync: false
    });

    $("#add-bn-structure").click(function () {
        $("#upload-bn-structure-div").removeClass("hide-element");
        $("#upload-bn-structure-div").show();
        $("#structure-1-text").removeClass("hide-element");
        $("#structure-1-text").show();
        $("#structure-2-text").removeClass("hide-element");
        $("#structure-2-text").show();
        $("#add-bn-structure").hide();
    });

    $("#upload-bn-structure-2").fileinput(
        {
            hideThumbnailContent: true, // hide image, pdf, text or other content in the thumbnail preview
            theme: "explorer",
            uploadUrl: "/morpho/ml_bn_load_only_structure/",
            fileActionSettings: {
                showDownload: false,
                showUpload: false,
                indicatorNew: '',
                removeIcon: '<i class="glyphicon glyphicon-trash fa-2x" aria-hidden="true"></i>',
                zoomIcon: '<i class="glyphicon glyphicon-zoom-in fa-2x"></i>',
            },
            previewClass: "file-input-small container-center",
            layoutTemplates: {
                main1: "<div class=\'container-center hide-element\'  id=\'upload-bn-structure-div\'>\n" +
                    "{preview}\n" +
                    "<div class=\'input-group {class} file-input-browse-small\'>\n" +
                    "   <div class=\'input-group-btn\ input-group-prepend file-input-button-small'>\n" +
                    "       {browse}\n" +
                    "       {upload}\n" +
                    "       {remove}\n" +
                    "   </div>\n" +
                    "   {caption}\n" +
                    "</div>" +
                    "</div>"
            },
            browseIcon: "<i class=\"glyphicon glyphicon-folder-open icon-margin-right\"></i> ",
            uploadExtraData: function (previewId, index) {
                var obj = {};
                obj['csrfmiddlewaretoken'] = csrf_token;
                obj['structure'] = 2;
                return obj;
            },
            uploadAsync: false
        });

    $('#upload-bn-structure-1').on('filebatchpreupload', function () {
        $("#select-group-graph").addClass("hide-element").hide();
        $("#select-nodes-graph-evidence-effect").addClass("hide-element").hide();
        $("#select-group-graph-evidence-effect").addClass("hide-element").hide();
    });

    $('#upload-bn-structure-1').on('filebatchuploadcomplete', function () {
        $("#upload-complete-label").text("Structure upload complete!");
        $("#continue-button-upload-bn").trigger("click");
    });

    $('#upload-bn-structure-2').on('filebatchuploadsuccess', function (event, data) {
        var placeholder_html = $("#section-structures-metrics-content-table");
        placeholder_html.empty();
        placeholder_html.append(data.response["score_results"]);
        placeholder_html.find('.datatable-new-ajax').each(function () {
            create_datatable($(this));
        });
        $("#section-structures-metrics").removeClass("hide-element").show();
    });

    $('#upload-bn-structure-2').on('filebatchuploadcomplete', function (event, files, extra) {
        $("#upload-complete-label").text("Structure 2 upload complete!");
        $("#continue-button-upload-bn").trigger("click");
        $("#select-graph-structure").removeClass("hide-element").show();
    });


    $('#selectize-bn-structure').on("change", function (e) {
        if ($("#model_name").val() !== "ml_probabilistic_clustering") {
            var structure_option_id = $(this).val();
            sigmajs_bn.graph.edges().forEach(function (edge) {
                if (parseInt(structure_option_id) === 3) {
                    if (edge.originalColor === sigmajs_bn_settings["color_common_edges"]) {
                        change_node_edge_color(edge, sigmajs_bn_settings["color_common_edges"], true);
                        edge.structure_selected = true;
                    } else {
                        hide_node_edge(edge);
                        edge.structure_selected = true;
                    }
                } else if (parseInt(structure_option_id) === 0) {
                    change_node_edge_color(edge, edge.originalColor, false);
                } else if (parseInt(structure_option_id) === edge.structure_id || edge.structure_id === 0) {
                    if (parseInt(structure_option_id) === 1) {
                        change_node_edge_color(edge, sigmajs_bn_settings["color_structure_1"], true);
                        edge.structure_selected = true;
                    } else {
                        change_node_edge_color(edge, sigmajs_bn_settings["color_structure_2"], true);
                        edge.structure_selected = true;
                    }
                } else {
                    hide_node_edge(edge);
                    edge.structure_selected = true;
                }
            });
            sigmajs_bn.refresh();
        } else {
            var structure_id = $(this).val();

            if (structure_id === "all") {
                $("#select-multi-structures").removeClass("hide-element").show();
                $("#select-bn-structure-common-edges").removeClass("hide-element").show();
            } else {
                $("#select-multi-structures").hide();
                $("#select-bn-structure-common-edges").hide();
            }

            reset_nodes_edges_color_prob_clustering();

            draw_evidences(structure_id);
        }
    });

    $('#selectize-bn-structure-common-edges').on("change", function (e) {
        var val_common_edges = $(this).val();
        var select_multi_structures = $("#selectize-multi-structures");
        var selectize_multi_structures = select_multi_structures[0].selectize;
        var vals_multi_structures = select_multi_structures.val();

        if (vals_multi_structures.length > 0 && val_common_edges !== "no-common") {
            selectize_clear_items(selectize_multi_structures);
        }

        reset_nodes_edges_color_prob_clustering();
    });

    if ($("#model_name").val() === "ml_probabilistic_clustering") {
        $('#selectize-multi-structures')[0].selectize.on("item_add", function (value, item) {
            var select_common_edges = $("#selectize-bn-structure-common-edges");
            var val_common_edges = select_common_edges.val();
            var selectize_bn_structure_common_edges = select_common_edges[0].selectize;

            if (val_common_edges !== "no-common") {
                selectize_bn_structure_common_edges.addItem("no-common");
            }
        });

        $('#selectize-multi-structures').on("change", function (value, item) {
            reset_nodes_edges_color_prob_clustering();
        });
    }

    $("#upload-bn-params-continuous").fileinput({
        hideThumbnailContent: true, // hide image, pdf, text or other content in the thumbnail preview
        theme: "explorer",
        uploadUrl: "/morpho/ml_bn_load_only_params_continuous/",
        fileActionSettings: {
            showDownload: false,
            showUpload: false,
            indicatorNew: '',
            removeIcon: '<i class="glyphicon glyphicon-trash fa-2x" aria-hidden="true"></i>',
            zoomIcon: '<i class="glyphicon glyphicon-zoom-in fa-2x"></i>',
        },
        previewClass: "file-input-small container-center",
        layoutTemplates: {
            main1: "<div class=\'container-center\'>\n" +
                "{preview}\n" +
                "<div class=\'input-group {class} file-input-browse-small\'>\n" +
                "   <div class=\'input-group-btn\ input-group-prepend file-input-button-small'>\n" +
                "       {browse}\n" +
                "       {upload}\n" +
                "       {remove}\n" +
                "   </div>\n" +
                "   {caption}\n" +
                "</div>" +
                "</div>"
        },
        browseIcon: "<i class=\"glyphicon glyphicon-folder-open icon-margin-right\"></i> ",
        uploadExtraData: function (previewId, index) {
            var obj = {}
            obj['csrfmiddlewaretoken'] = csrf_token;
            return obj;
        },
        uploadAsync: false
    });

    $('#upload-bn-params-continuous').on('filebatchpreupload', function (event, data) {

    });

    $('#upload-bn-params-continuous').on('filebatchuploadcomplete', function (event, files, extra) {
        $("#upload-complete-label").text("Parameters upload complete!")
        $("#continue-button-upload-bn").trigger("click");
    });

    $("#upload-bn").fileinput({
        hideThumbnailContent: true, // hide image, pdf, text or other content in the thumbnail preview
        theme: "explorer",
        uploadUrl: "/morpho/ml_upload_bn_file/",
        fileActionSettings: {
            showDownload: false,
            showUpload: false,
            indicatorNew: '',
            removeIcon: '<i class="glyphicon glyphicon-trash fa-2x" aria-hidden="true"></i>',
            zoomIcon: '<i class="glyphicon glyphicon-zoom-in fa-2x"></i>',
        },
        previewClass: "file-input-small container-center",
        layoutTemplates: {
            main1: "<div class=\'container-center\'>\n" +
                "{preview}\n" +
                "<div class=\'input-group {class} file-input-browse-small\'>\n" +
                "   <div class=\'input-group-btn\ input-group-prepend file-input-button-small'>\n" +
                "       {browse}\n" +
                "       {upload}\n" +
                "       {remove}\n" +
                "   </div>\n" +
                "   {caption}\n" +
                "</div>" +
                "</div>"
        },
        uploadExtraData: function (previewId, index) {
            var obj = {}

            obj['csrfmiddlewaretoken'] = csrf_token;
            obj['classes-bn-file'] = $("#select-classes-bn-file").val();
            obj['bn-name'] = $("#bn-input-name").val();

            return obj;
        },
        uploadAsync: false
    });

    $('#upload-bn').on('filebatchpreupload', function (event, data) {

    });

    $('#upload-bn').on('filebatchuploadcomplete', function (event, files, extra) {
        $("#upload-complete-label").text("Upload complete! Click in the Continue button at the bottom.")
    });

    $("#upload-bn-additional-parameters").fileinput({
        hideThumbnailContent: true, // hide image, pdf, text or other content in the thumbnail preview
        theme: "explorer",
        uploadUrl: "/morpho/ml_upload_bn_additional_parameters/",
        fileActionSettings: {
            showDownload: false,
            showUpload: true,
            indicatorNew: '',
            removeIcon: '<i class="glyphicon glyphicon-trash fa-2x" aria-hidden="true"></i>',
            zoomIcon: '<i class="glyphicon glyphicon-zoom-in fa-2x"></i>',
        },
        previewClass: "file-input-small container-center",
        layoutTemplates: {
            main1: "<div class=\'container-center\'>\n" +
                "{preview}\n" +
                "<div class=\'input-group {class} file-input-browse-small\'>\n" +
                "   <div class=\'input-group-btn\ input-group-prepend file-input-button-small'>\n" +
                "       {browse}\n" +
                "       {upload}\n" +
                "       {remove}\n" +
                "   </div>\n" +
                "   {caption}\n" +
                "</div>" +
                "</div>"
        },
        browseIcon: "<i class=\"glyphicon glyphicon-folder-open icon-margin-right\"></i> ",
        uploadExtraData: function (previewId, index) {
            var obj = {}
            obj['model_name'] = $("#model_name").val();
            obj['csrfmiddlewaretoken'] = csrf_token;
            return obj;
        },
        uploadAsync: false
    });

    $('#upload-bn-additional-parameters').on('filebatchuploadcomplete', function (event, files, extra) {
        if ($("#model_name").val() === "ml_probabilistic_clustering" && $("#datatable_available").val() === "True") {
            init_update_selectize_datatable_cols_groups();
        }
        reload_sigmajs_graph();
        $("#success-upload-additional-parameters-bn").removeClass("hide-element");
        $("#success-upload-additional-parameters-bn").show();
    });

    $("#checkbox-dataset-discretize").change(function (e) {
        var is_checked = $(this).is(":checked");
        if (is_checked) {
            $("#discretize-methods").removeClass("hide-element");
            $("#discretize-methods").show();
        } else {
            $("#discretize-methods").hide();
            $(".discretize-parameters").hide(); //Close old methods parameters input
        }
    });

    $("#select-dataset-discretize-method").on("change", function (e) {
        var method_chosen = $(this).val();
        var placeholder_method_paremeters = $("#discretize-parameters-" + method_chosen);

        $(".discretize-parameters").hide(); //Close old methods parameters input
        placeholder_method_paremeters.removeClass("hide-element").show();
    });

    $(".dataset-features").selectize({});

    $("#select-dataset").on("change", function (e) {
        var values_chosen = $(this).val();
        if (values_chosen.length !== 0 && values_chosen !== "") {
            var value_str = "";
            var choose_value_elem = $('#' + $(this).attr("choose_value_elem"));
            var selectizes_value_elem = $('.' + $(this).attr("selectize_value_elem"));
            var waiting_div_elem = $('#' + $(this).attr("waiting_div_elem"));
            choose_value_elem.removeClass("hide-element");
            choose_value_elem.show();

            selectizes_value_elem.each(function (index) {
                $(this).selectize()[0].selectize.destroy();
                $(this).empty();
            });
            var data_client_json = {
                "datasets_names": values_chosen,
                "include_ids": false,
                "model_name": $("#model_name").val()
            };
            data_client_json = JSON.stringify(data_client_json)
            var data_send = {
                "csrfmiddlewaretoken": csrf_token,
                "data_client_json": data_client_json
            };
            $.ajax({
                type: "POST",
                url: "/morpho/get_features_datasets/",
                data: data_send,
                dataType: 'json',
                beforeSend: function () {
                    waiting_div_elem.removeClass("hide-element");
                    waiting_div_elem.show();
                },
                success: function (data) {
                    waiting_div_elem.hide();
                    $("#error-upload-bn").empty();

                    var values = data["values"];
                    var maxItems = values.length;
                    var maxOptions = values.length;
                    var items = values.map(function (x) {
                        return {item: x};
                    });

                    selectizes_value_elem.selectize({
                        maxItems: maxItems,
                        maxOptions: maxOptions,
                        labelField: "item",
                        valueField: "item",
                        searchField: "item",
                        options: items,
                    });

                },
                error: function () {
                    waiting_div_elem.hide();
                    $("#error-upload-bn").val("Server error");
                }
            });
        }
    });


    $("#continue-button-upload-dataset").click(function (e) {
        var bn_name = $("#bn-input-name").val();
        var dataframes_names = $("#select-dataset").val();
        var dataframes_features = {};
        var select_dataframes_features = $("#select-dataset-features");
        var selectize_dataframes_features = select_dataframes_features[0].selectize;
        var dataframes_features_all = select_dataframes_features.attr("selected-all");
        var dataframes_features_classes = $("#select-dataset-classes").val();
        var discretize = false;
        var discretize_method = "";
        var discretize_method_parameters = {};
        if ($("#checkbox-dataset-discretize").is(":checked")) {
            discretize = true;
            discretize_method = $("#select-dataset-discretize-method").val();
            $(".discretize-parameters:visible .discretize-parameter").each(function () {
                var parameter_name = $(this).attr("id");
                var parameter_value = $(this).val();
                discretize_method_parameters[parameter_name] = parameter_value;
            });
        }

        dataframes_features["values"] = $("#select-dataset-features").val();
        dataframes_features["selected_all"] = false;
        if (dataframes_features_all === "true") {
            dataframes_features["selected_all"] = true;
        }

        var progress_elem_html_id = "progress-upload-dataset-bn";
        var progress_elem_html = $("#" + progress_elem_html_id);
        var waiting_div_elem = $("#waiting-continue-upload-dataset-bn");
        var error_elem = $("#error-continue-upload-dataset-bn");
        var success_elem = $("#success-continue-upload-dataset-bn");
        var next_section_elem = $("#section-learn-structure-bn");

        var data_client_json = {
            "bn_name": bn_name,
            "datasets_names": dataframes_names,
            "dataframes_features": dataframes_features,
            "dataframes_features_classes": dataframes_features_classes,
            "discretize": discretize,
            "discretize_method": discretize_method,
            "discretize_method_parameters": discretize_method_parameters,
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json)
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_upload_dataset/",
            data: data_send,
            dataType: 'json',
            beforeSend: function () {
                waiting_div_elem.removeClass("hide-element");
                waiting_div_elem.show();
                progress_elem_html.show();
            },
            success: function (data_task) {
                var data_processed = run_worker(data_task, progress_elem_html_id, function (data_processed) {
                    waiting_div_elem.hide();
                    if (data_processed["error"]) {
                        error_elem.removeClass("hide-element");
                        error_elem.show();
                        error_elem.text("Error: " + data_processed["error_message"]);
                        success_elem.hide();
                    } else {
                        error_elem.empty();
                        success_elem.removeClass("hide-element");
                        success_elem.show();
                        next_section_elem.removeClass("hide-element");
                        next_section_elem.show();
                    }
                })
            },
            error: function () {
                waiting_div_elem.hide();
                success_elem.show();
                error_elem.text("Server error");
            }
        });
    });

    $("#continue-button-upload-bn").click(function (e) {
        var bn_features_classes = $("#select-classes-bn-file").val();
        var waiting_div_elem = $("#waiting-continue-upload-dataset-bn");
        var success_elem = $("#success-continue-upload-dataset-bn");
        var error_elem = $("#error-continue-upload-dataset-bn");
        var next_section_elem = $("#section-draw-bn");
        var next_section_elem_2 = $("#section-export-bn");

        var data_client_json = {
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json)
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_upload_bn/",
            data: data_send,
            dataType: 'json',
            beforeSend: function () {
                waiting_div_elem.removeClass("hide-element");
                waiting_div_elem.show();
            },
            success: function (data) {
                waiting_div_elem.hide();
                if (data["error"]) {
                    error_elem.text(data["error_message"]);
                    success_elem.hide();
                } else {
                    error_elem.empty();
                    success_elem.removeClass("hide-element");
                    success_elem.show();
                    next_section_elem.removeClass("hide-element");
                    next_section_elem.show();
                    next_section_elem_2.removeClass("hide-element");
                    next_section_elem_2.show();

                    $('html, body').animate({
                        scrollTop: $("#section-draw-bn").offset().top
                    }, 1000);

                    draw_bayesian_network_sigma(data["graph_sigmajs"], data["sigmajs_default_settings"], data["nodes"], data["additional_discrete_features"]);
                }
            },
            error: function () {
                waiting_div_elem.hide();
                success_elem.hide();
                error_elem.text("Server error");
            }
        });
    });

    $("#select-structure-algorithm").on("change", function (e) {
        var algorithm_chosen = $(this).val();
        var placeholder_algorithm_paremeters = $("#structure-learning-parameters-" + algorithm_chosen);

        $("#container-continue-learn-structure").removeClass("hide-element");
        $("#container-continue-learn-structure").show();

        $("#section-structure-learning-parameters").removeClass("hide-element");
        $("#section-structure-learning-parameters").show();
        $(".structure-learning-algorithm-parameters").hide(); //Close old algorithm parameters input
        placeholder_algorithm_paremeters.show();
    });


    $("#continue-learn-structure").click(function (e) {
        var structure_algorithm = $("#select-structure-algorithm").val();
        var structure_algorithm_parameters = {};

        var waiting_div_elem = $("#waiting-continue-learn-structure");
        var progress_elem_html_id = "progress-bn-learn-structure";
        var progress_elem_html = $("#" + progress_elem_html_id);
        var error_elem = $("#error-continue-learn-structure");
        var success_elem = $("#success-continue-learn-structure");
        var this_section = $("#section-learn-structure-bn");
        var next_section_1_elem = $("#section-parameters-learning-bn");
        var next_section_2_elem = $("#section-draw-bn");
        var next_section_3_elem = $("#section-export-bn");


        $(".structure-learning-algorithm-parameters:visible .structure-learning-parameter").each(function () {
            var parameter_name = $(this).attr("id");
            if (parameter_name !== undefined) {
                var parameter_value = $(this).val();
                structure_algorithm_parameters[parameter_name] = parameter_value;
            }
        });

        var data_client_json = {
            "structure_algorithm": structure_algorithm,
            "structure_algorithm_parameters": structure_algorithm_parameters,
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json)
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_learn_structure/",
            data: data_send,
            dataType: 'json',
            beforeSend: function () {
                waiting_div_elem.removeClass("hide-element");
                waiting_div_elem.show();
                progress_elem_html.show();
                error_elem.hide();
            },
            success: function (data_task) {
                var data_processed = run_worker(data_task, progress_elem_html_id, function (data_processed) {
                    waiting_div_elem.hide();
                    if (data_processed["error"]) {
                        error_elem.removeClass("hide-element").show();
                        error_elem.text("Error: " + data_processed["error_message"]);
                        success_elem.hide();
                    } else {
                        error_elem.empty();
                        error_elem.hide();
                        success_elem.removeClass("hide-element");
                        success_elem.show();
                        this_section.removeClass("last-section-page");
                        next_section_1_elem.removeClass("hide-element");
                        next_section_1_elem.show();
                        next_section_2_elem.removeClass("hide-element");
                        next_section_2_elem.show();
                        next_section_3_elem.removeClass("hide-element");
                        next_section_3_elem.show();
                        draw_bayesian_network_sigma(data_processed["graph_sigmajs"], data_processed["sigmajs_default_settings"], data_processed["nodes"], data_processed["additional_discrete_features"]);
                    }
                });
            },
            error: function (e) {
                waiting_div_elem.hide();
                success_elem.hide();
                error_elem.removeClass("hide-element").show();
                next_section_1_elem.hide();
                next_section_2_elem.hide();
                error_elem.text("Error: check if the selected columns have NaN or empty values");
            }
        });
    });

    $("#select-parameters-algorithm").on("change", function (e) {
        var algorithm_chosen = $(this).val();
        var placeholder_algorithm_paremeters = $("#parameters-learning-parameters-" + algorithm_chosen);

        $("#container-continue-learn-parameters").removeClass("hide-element");
        $("#container-continue-learn-parameters").show();

        $("#section-parameters-learning-parameters").removeClass("hide-element");
        $("#section-parameters-learning-parameters").show();
        $(".parameters-learning-algorithm-parameters").hide(); //Close old algorithm parameters input
        placeholder_algorithm_paremeters.show();
    });

    $("#bayesianEstimation_prior").on("change", function (e) {
        var prior_chosen = $(this).val();
        if (prior_chosen === "BDeu") {
            $("#bayesianEstimation_set_equivalent_size").removeClass("hide-element");
            $("#bayesianEstimation_set_equivalent_size").show();
        } else {
            $("#bayesianEstimation_set_equivalent_size").hide();
        }
    });

    $("#continue-learn-parameters").click(function (e) {
        var parameters_algorithm = $("#select-parameters-algorithm").val();
        var algorithm_parameters = {};

        var progress_elem_html_id = "progress-bn-learn-parameters";
        var progress_elem_html = $("#" + progress_elem_html_id);
        var waiting_div_elem = $("#waiting-continue-learn-parameters");
        var error_elem = $("#error-continue-learn-parameters");
        var success_elem = $("#success-continue-learn-parameters");
        var this_section = $("#section-learn-parameters-bn");
        var next_section_1_elem = $("#section-draw-bn");
        var next_section_2_elem = $("#section-export-bn");


        $(".parameters-learning-algorithm-parameters:visible .parameters-learning-parameter").each(function () {
            var parameter_name = $(this).attr("id");
            if (parameter_name !== undefined) {
                var parameter_value = $(this).val();
                algorithm_parameters[parameter_name] = parameter_value;
            }
        });

        var data_client_json = {
            "parameters_algorithm": parameters_algorithm,
            "algorithm_parameters": algorithm_parameters,
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json)
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_learn_parameters/",
            data: data_send,
            dataType: 'json',
            beforeSend: function () {
                waiting_div_elem.removeClass("hide-element");
                waiting_div_elem.show();
                progress_elem_html.show();
                error_elem.hide();
            },
            success: function (data_task) {
                var data_processed = run_worker(data_task, progress_elem_html_id, function (data_processed) {
                    waiting_div_elem.hide();
                    if (data_processed["error"]) {
                        error_elem.removeClass("hide-element");
                        error_elem.show();
                        error_elem.text("Error: " + data_processed["error_message"]);
                        success_elem.hide();
                    } else {
                        error_elem.empty();
                        error_elem.hide();
                        success_elem.removeClass("hide-element");
                        success_elem.show();
                        this_section.removeClass("last-section-page");
                        next_section_1_elem.removeClass("hide-element");
                        next_section_1_elem.show();
                        next_section_2_elem.removeClass("hide-element");
                        next_section_2_elem.show();
                        draw_bayesian_network_sigma(data_processed["graph_sigmajs"], data_processed["sigmajs_default_settings"], data_processed["nodes"], data_processed["additional_discrete_features"]);
                    }
                });
            },
            error: function () {
                waiting_div_elem.hide();
                success_elem.hide();
                error_elem.removeClass("hide-element").show();
                next_section_1_elem.hide();
                next_section_2_elem.hide();
                error_elem.text("Error: check if the selected columns have NaN or empty values");
            }
        });
    });


    $("#select-export-bn-formats").on("change", function (e) {
        $("#section-continue-export-bn").removeClass("hide-element");
        $("#section-continue-export-bn").show();
    });

    $("#continue-export-bn").click(function (e) {
        var form = $("#form-export-bn");
        var file_format = $("#select-export-bn-formats").val();
        var waiting_div_elem = $("#waiting-operation-export-bn");

        waiting_div_elem.removeClass("hide-element");
        waiting_div_elem.show();

        $("#file-format-export-bn").val(file_format);
        form.submit();

        waiting_div_elem.hide();
    });

    if ($("#model_name").val() === "ml_probabilistic_clustering") {
        init_sliders_glasso();
    }

    $("#checkbox-layout-forceatlas2-client").on("change", function () {
        var checked = $(this).is(":checked");
        if (checked) {
            sigmajs_bn.startForceAtlas2({});
            setTimeout(function () {
                sigmajs_bn.stopForceAtlas2();
            }, 500000);
        } else {
            sigmajs_bn.stopForceAtlas2();
        }
    });

    $(".bn-options-change-layout-server").click(function (e) {
        var layout_name = $(this).attr("layout-name");
        var additional_params = {};

        if (layout_name === "image") {
            additional_params["layout-image-url"] = $("#layout-image-url").val();
            additional_params["layout-image-threshold"] = $("#layout-image-threshold").val();
        }

        change_layout_sigmajs(layout_name, additional_params);
    });


    function change_layout_sigmajs(layout_name, additional_params) {
        var notify_loading = print_notify_message(permanent = true, type = "info", icon = 'fa fa-spinner fa-spin icon-margin-right', title = "Loading layout", message = "Loading " + layout_name + " layout", delay = 4000, additional_message = "");

        var data_client_json = {
            "layout_name": layout_name,
            "additional_params": additional_params,
            "model_name": $("#model_name").val()
        };

        data_client_json = JSON.stringify(data_client_json);
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_change_layout/",
            data: data_send,
            dataType: 'json',
            beforeSend: function () {
                //clear_sigmajs_instance(sigmajs_bn);
                sigmajs_bn.stopForceAtlas2();
            },
            success: function (data) {
                notify_loading.close();

                if (data["error"]) {
                    var notify_error = print_notify_message(permanent = false, type = "danger", icon = 'fa fa-spinner fa-spin icon-margin-right', title = "Error loading layout", message = data["error_message"], delay = 3000);
                } else {
                    sigmajs_bn_settings["active_layout"] = layout_name;
                    var nodes_pos = data["nodes_pos"];
                    var edges = data["edges"];

                    for (let node_id in nodes_pos) {
                        var node_sigmajs = sigmajs_bn.graph.getNode(node_id);
                        node_sigmajs.x = nodes_pos[node_id][0];
                        node_sigmajs.y = nodes_pos[node_id][1];
                    }

                    sigmajs_bn.graph.edges().forEach(function (edge) {
                        sigmajs_bn.graph.dropEdge(edge.id);
                    });

                    for (let edge_id in edges) {
                        sigmajs_bn.graph.addEdge(edges[edge_id]);
                    }

                    if ($("#model_name").val() === "ml_probabilistic_clustering") {
                        initialize_selectize_structures(sigmajs_bn_settings["structures_info"]);
                    }

                    show_hide_arrows();

                    sigmajs_bn.refresh(); // Ask sigma to draw it
                }
            },
            error: function () {
                notify_loading.close();
                var notify_error = print_notify_message(permanent = false, type = "danger", icon = 'fa fa-spinner fa-spin icon-margin-right', title = "Error loading layout", message = "Server error", delay = 3000);
            }
        });
    }

    $("#checkbox-show-hide-labels").on("change", function (e) {
        var show_labels = $(this).is(":checked");
        sigmajs_bn.settings('drawLabels', show_labels);
        sigmajs_bn.refresh(); // Ask sigma to draw it
    });

    $("#checkbox-show-hide-arrows").on("change", function (e) {
        show_hide_arrows();
    });

    $("#checkbox-allow-drag-drop").on("change", function (e) {
        var allow_drag_drop = $(this).is(":checked");
        if (allow_drag_drop) {
            var dragListener = sigma.plugins.dragNodes(sigmajs_bn, sigmajs_bn.renderers[0]);
        } else {
            sigma.plugins.killDragNodes(sigmajs_bn);
        }
    });


    $("#checkbox-on-click-group-hide-others").on("change", function (e) {
        var hide = $(this).is(":checked");

        sigmajs_bn.graph.nodes().forEach(function (n) {
            if (!hide && n.hidden) {
                hide_node_edge(n);
            } else if (n.color === sigmajs_bn_settings["color_hidden"]) {
                n.hidden = hide;
            }
        });

        sigmajs_bn.graph.edges().forEach(function (e) {
            if (!hide && e.hidden) {
                hide_node_edge(e);
            } else if (e.color === sigmajs_bn_settings["color_hidden"]) {
                e.hidden = hide;
            }
        });

        sigmajs_bn.refresh();
    });


    $("#checkbox-on-click-group-show-neighbors").on("change", function (e) {
        if (sigmajs_bn_settings["active_category"] !== "") {
            var notify_loading = undefined;
            get_all_nodes_in_category(sigmajs_bn, sigmajs_bn_settings["active_category"],
                "render_nodes_in_category", notify_loading);
        }
    });

    $("#bn-options-view-toggle-autoRescale").click(function (e) {
        var autoRescale = true;
        if (sigmajs_bn.settings('autoRescale') == true) {
            autoRescale = false;
        }
        sigmajs_bn.settings('autoRescale', autoRescale);
        sigmajs_bn.refresh(); // Ask sigma to draw it
    });

    $("#bn-options-view-full-scren").click(function (e) {
        var bn_section_elem = $("#section-draw-bn")[0];
        var bn_graph_elem = $("#bayesian-network-graph");

        if (screenfull.enabled) {
            screenfull.toggle(bn_section_elem);

            //Change buttons layout to fit in the full screen
            $(".dropup").addClass("dropdown").removeClass("dropup");
            var height_section = $("#section-draw-bn").height();
            bn_graph_elem.css("height", height_section + 60);
        }
    });


    $(".export-bn-button").click(function (e) {
        var format = $(this).attr("export-format");

        if (format == "svg") {
            var output_file = sigmajs_bn.toSVG({download: true, filename: 'bn.svg', size: 1000});
        }
    });

    $("#bn-options-view-reload-graph").click(function (e) {
        reload_sigmajs_graph();
    });

    $('#modal-set-evidence-group').on('shown.bs.modal', function (event) {
        $('#modal-set-evidence-group').attr("group", event.relatedTarget.id);
    });

    $("#confirm-evidence-group").click(function (e) {
        var notify_loading = print_notify_message(permanent = true, type = "info",
            icon = 'fa fa-spinner fa-spin icon-margin-right', title = "Setting evidences",
            message = "Setting evidences for selected nodes.", delay = 0,
            additional_message = "", position = "bottom-right");

        var group = $('#modal-set-evidence-group').attr('group');
        if (group === "set-evidence-group-1") {
            var select_multi_nodes_bn = $("#select-multi-nodes-bn");
            var selected_nodes_ids = select_multi_nodes_bn.val();
            var evidence_group_value = $("#evidence-group-value");
            var new_evidence_value = evidence_group_value.val();
            $("#node-parameters-plot-copy").remove();
            set_nodes_evidences(selected_nodes_ids, new_evidence_value, {}, true, callback_function = function () {
                notify_loading.close();
            });
        } else if (group === "set-evidence-group-2") {
            var select_category_input = $("#select-category-bn");
            var category_id = select_category_input.val();
            get_all_nodes_in_category(null, category_id, "set_nodes_evidences", notify_loading);
        }
        $('#modal-set-evidence-group').attr("group", "0");

    });

    $("#clear-evidence-group").click(function (e) {
        var select_multi_nodes_bn = $("#select-multi-nodes-bn");
        var selected_nodes_ids = select_multi_nodes_bn.val();
        $("#node-parameters-plot-copy").remove();
        clear_nodes_evidences(selected_nodes_ids, {}, true);
    });

    $("#clear-evidence-group-2").click(function (e) {
        var select_category_input = $("#select-category-bn");
        var category_id = select_category_input.val();
        get_all_nodes_in_category(null, category_id, "clear_nodes_evidences");
    });

    $('#cp3').colorpicker({
        color: '#000000',
        format: 'rgb'
    });

    $("#confirm-create-custom-group").click(function (e) {
        var select_multi_nodes_bn = $("#select-multi-nodes-bn");
        var selected_nodes_ids = select_multi_nodes_bn.val();
        var custom_group_name = $("#name-custom-group");
        custom_group_name = custom_group_name.val();
        var custom_group_color = $("#color-picker");
        custom_group_color = custom_group_color.val();
        var custom_group_nodes = {};
        selected_nodes_ids.forEach(function (node) {
            custom_group_nodes[node] = {};
        });

        if (custom_group_name === "") {
            e.preventDefault();
            $("#create-custom-group-error").removeClass("hide-element").show();
        } else {
            $("#name-custom-group").val("");
            $("#create-custom-group-error").hide();
            $("#modal-create-custom-group").modal('hide');

            var data_client_json = {
                "custom_group": custom_group_nodes,
                "group_name": custom_group_name,
                "group_color": custom_group_color,
                "model_name": $("#model_name").val()
            };
            data_client_json = JSON.stringify(data_client_json);

            var data_send = {
                "csrfmiddlewaretoken": csrf_token,
                "data_client_json": data_client_json
            };
            $.ajax({
                type: "POST",
                url: "/morpho/ml_bn_create_custom_group/",
                data: data_send,
                dataType: 'json',
                success: function (data) {
                    if (!$.isEmptyObject(data["additional_parameters"])) {
                        set_additional_discrete_features(data["additional_parameters"]);
                        sigmajs_bn.refresh();
                    }
                }, error: function () {
                }
            });
        }
    });

    function get_edges_between_nodes(selected_nodes_ids) {
        reset_edges_color(sigmajs_bn);
        var data_client_json = {
            "nodes_ids": selected_nodes_ids,
            "structure_id": sigmajs_bn_settings["active_structure"],
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json);
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_get_edges_between_nodes/",
            data: data_send,
            dataType: 'json',
            success: function (data) {
                var selected_nodes = data["nodes_selection"];
                var num_edges_selected = 0;
                sigmajs_bn.graph.edges().forEach(function (e) {
                    if (!e.selected) {
                        var source = e.source;
                        var target = e.target;
                        var highlight_edge = false;
                        var valid_edge = false;

                        if (selected_nodes.indexOf(source) >= 0 && selected_nodes.indexOf(target) >= 0) {
                            valid_edge = true;
                        }

                        if (valid_edge) {
                            change_node_edge_color(e, e.originalColor, false);
                            e.selected = true;
                            highlight_edge = true;
                            num_edges_selected++;
                        }
                        if (!highlight_edge) {
                            hide_node_edge(e);
                        }
                    } else {
                        num_edges_selected++;
                    }
                });

                $("#network-stats-num-edges-selected").text(num_edges_selected);

                if ($("#model_name").val() === "ml_probabilistic_clustering") {
                    reset_edges_color_prob_clustering();
                }

                sigmajs_bn.refresh();
            },
            error: function (data) {
            }
        });
    }

    $("#show-group-nodes-edges").click(function (e) {
        var select_multi_nodes_bn = $("#select-multi-nodes-bn");
        var selected_nodes_ids = select_multi_nodes_bn.val();
        var nodeId = "";
        var selection = "";
        render_x_nodes(selected_nodes_ids, sigmajs_bn, selection, nodeId, sigmajs_bn_settings["highlight_node_color"]);
        get_edges_between_nodes(selected_nodes_ids);
    });


    if (screenfull.enabled) {
        screenfull.on('change', function (e) {
            if (!screenfull.isFullscreen) {
                $(".dropdown").addClass("dropup").removeClass("dropdown");
                var bn_graph_elem = $("#bayesian-network-graph");
                bn_graph_elem.css("height", sigmajs_bn_settings["display_original_height"]);
            }
        });
    }


    $("#bn-options-clean-all-notifications").click(function (e) {
        $.notifyClose();
    });

    function get_category_values_in_group(group_id) {
        var data_client_json = {
            "group_id": group_id,
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json)
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_category_values_in_group/",
            data: data_send,
            dataType: 'json',
            success: function (data) {
                //Search groups input:
                $("#select-category-graph").removeClass("hide-element");
                $("#select-category-graph").show();
                if (data["categories_in_group"].length > 0) {
                    var select_category_input = $("#select-category-bn");
                    select_category_input.selectize()[0].selectize.destroy();
                    select_category_input.empty();
                    var items = data["categories_in_group"].map(function (x) {
                        return {item: x};
                    });
                    items.push({"item": "All categories"});
                    select_category_input.selectize({
                        maxItems: 1,
                        maxOptions: data["categories_in_group"].length + 1,
                        labelField: "item",
                        valueField: "item",
                        searchField: "item",
                        options: items,
                    });
                    select_category_input.on("change", function (e) {
                        var category_chosen_id = $(this).val();
                        if (category_chosen_id == "All categories") {
                            reset_nodes_edges_color(sigmajs_bn);
                            $("#button-set-evidence-group").hide();
                            $("#button-clear-evidence-group").hide();
                            sigmajs_bn_settings["active_category"] = "";
                            get_and_render_all_nodes_by_group(sigmajs_bn, sigmajs_bn_settings["active_group"]);
                        } else {
                            get_all_nodes_in_category(sigmajs_bn, category_chosen_id, "render_nodes_in_category");
                            $("#button-set-evidence-group").removeClass("hide-element").show();
                            $("#button-clear-evidence-group").removeClass("hide-element").show();
                            $("#set-evidence-group-2").text("Set evidence in " + category_chosen_id);
                            $("#clear-evidence-group-2").text("Clear evidence in " + category_chosen_id);
                        }
                    });
                } else {
                    reset_nodes_edges_color(sigmajs_bn);
                    $("#select-category-graph").hide();
                }
            },
            error: function () {
            }
        });
    }

    $("#check-dseparation-nodes").click(function (e) {
        var start_nodes = $("#select-start-nodes-bn").val();
        var end_nodes = $("#select-end-nodes-bn").val();
        var observed_nodes = $("#select-observed-nodes-bn").val();
        var data_client_json = {
            "start_nodes": start_nodes,
            "end_nodes": end_nodes,
            "observed_nodes": observed_nodes
        };
        data_client_json = JSON.stringify(data_client_json)
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_is_dseparated/",
            data: data_send,
            dataType: 'json',
            success: function (data) {
                if ($('[data-notify="container"]').length > 3) {
                    $.notifyClose();
                }
                if (data["is_dseparated"] === true) {
                    print_notify_message(permanent = true, type = "info",
                        icon = 'fa fa-info icon-margin-right', title = "D-separation",
                        message = "D-separation confirmed. Group 1 = [" + start_nodes + "], Group 2 = [" + end_nodes + "], Observed nodes = [" + observed_nodes + "]", delay = 0,
                        additional_message = "", position = "bottom-right");
                } else {
                    print_notify_message(permanent = true, type = "info",
                        icon = 'fa fa-info icon-margin-right', title = "D-separation",
                        message = "D-separation unconfirmed. Group 1 = [" + start_nodes + "], Group 2 = [" + end_nodes + "], Observed nodes = [" + observed_nodes + "]", delay = 0,
                        additional_message = "", position = "bottom-right");
                }
            },
            error: function () {
            }
        });
    });

    $("#get-dseparated-nodes").click(function (e) {
        var start_nodes = $("#select-start-nodes-bn").val();

        var data_client_json = {
            "start_nodes": start_nodes
        };
        data_client_json = JSON.stringify(data_client_json)
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_get_dseparated_nodes/",
            data: data_send,
            dataType: 'json',
            success: function (data) {
                if ($('[data-notify="container"]').length > 3) {
                    $.notifyClose();
                }
                var print_message = "";
                for (var key in data["reachable_nodes"]) {
                    print_message += "<strong>" + key + "</strong> = [" + data["reachable_nodes"][key] + "]\n"
                }
                print_notify_message(permanent = true, type = "info",
                    icon = 'fa fa-info icon-margin-right', title = "Reachable nodes from each node",
                    message = print_message, delay = 0,
                    additional_message = "", position = "bottom-right");
            },
            error: function () {
            }
        });
    });


    function draw_bayesian_network_sigma(graph_sigma_js, sigmajs_default_settings, nodes, additional_discrete_features) {
        $("#section-draw-bn").removeClass("hide-element");
        $("#section-draw-bn").show();
        $("#bayesian-network-graph").empty();

        clear_sigmajs_instance(sigmajs_bn);
        init_sigmajs_plugins_before_creation(sigmajs_default_settings);

        initialize_sigmajs_bn_instance(sigmajs_default_settings);
        sigmajs_bn.graph.read(graph_sigma_js);

        init_sigmajs_plugins_after_creation(sigmajs_default_settings, nodes, additional_discrete_features);

        show_hide_arrows();
        draw_evidences();

        sigmajs_bn.refresh(); // Ask sigma to draw it
    }

    function initialize_sigmajs_bn_instance(sigmajs_default_settings) {
        sigmajs_bn = new sigma(
            {
                renderer: {
                    container: document.getElementById(sigmajs_bn_settings["instance_id"]),
                    type: sigmajs_bn_settings["drawing_engine"],
                },
                settings: sigmajs_default_settings,
            }
        );
        resetCameraSigmajs(sigmajs_bn);
    }

    function clear_sigmajs_instance(sigma_js_instance) {
        var placeholder_html_graph = $("#" + sigmajs_bn_settings["instance_id"]);
        placeholder_html_graph.empty();
        reset_sigmajs_buttons();

        if (sigma_js_instance.graph !== undefined && sigma_js_instance.graph.nodesArray !== undefined) {
            //this gets rid of all the nodes and edges
            sigma_js_instance.graph.clear();
            //this gets rid of any methods you've attached to s.
            sigma_js_instance.graph.kill();
        }
    }

    function draw_evidences(structure_id) {
        var data_client_json = {
            "structure_id": structure_id,
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json);
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_get_nodes_evidences/",
            data: data_send,
            dataType: 'json',
            success: function (data) {
                var nodes_evidences = data["nodes_evidences"];

                sigmajs_bn.graph.nodes().forEach(function (node) {
                    if (nodes_evidences.indexOf(node.id) >= 0) {
                        change_node_edge_color(node, sigmajs_bn_settings["evidence_set_color"], false);
                        node.evidence_set = true;
                        node.selected_important = true;
                    } else {
                        node.evidence_set = false;
                        node.selected_important = false;
                    }
                });
                sigmajs_bn.refresh();
            },
            error: function () {
            }
        });
    }

    function reset_sigmajs_buttons() {
        $(".checkbox-option-bn").prop("checked", false);
        $(".button-binded-bn").off();
    }


    function show_hide_arrows() {
        var show_arrows = $("#checkbox-show-hide-arrows").is(":checked");
        sigmajs_bn_settings["show_arrows"] = show_arrows;
        sigmajs_bn.graph.edges().forEach(function (e) {
            if (show_arrows) {
                e.type = "arrow";
            } else {
                e.type = "line";
            }

        });
        sigmajs_bn.refresh(); // Ask sigma to draw it
    }

    function init_sigmajs_plugins_before_creation(sigmajs_default_settings) {
        if (!sigma.classes.graph.hasMethod('neighbors')) {
            sigma.classes.graph.addMethod('neighbors', function (nodeId) {
                var k,
                    neighbors = {},
                    index = this.allNeighborsIndex[nodeId] || {};

                for (k in index)
                    neighbors[k] = this.nodesIndex[k];

                return neighbors;
            });
        }
        if (!sigma.classes.graph.hasMethod('getNode')) {
            sigma.classes.graph.addMethod('getNode', function (nodeId) {
                return this.nodesIndex[nodeId];
                ;
            });
        }

        sigmajs_bn_settings["original_color"] = sigmajs_default_settings["nodeOriginalColor"];
        sigmajs_bn_settings["num_nodes"] = sigmajs_default_settings["num_nodes"];
        sigmajs_bn_settings["num_edges"] = sigmajs_default_settings["num_edges"];


        if (sigmajs_bn_settings["width_slider"].noUiSlider) {
            sigmajs_bn_settings["width_slider"].noUiSlider.destroy();
        }
        noUiSlider.create(sigmajs_bn_settings["width_slider"], {
            start: [sigmajs_default_settings["width_original"]],
            connect: true,
            range: {
                'min': sigmajs_default_settings["width_original"],
                'max': sigmajs_default_settings["width_max"],
            },
            step: sigmajs_default_settings["width_step"],
            pips: {
                mode: 'count',
                values: sigmajs_default_settings["width_number_options"],
                density: 1
            }
        });

        if (sigmajs_bn_settings["height_slider"].noUiSlider) {
            sigmajs_bn_settings["height_slider"].noUiSlider.destroy();
        }
        noUiSlider.create(sigmajs_bn_settings["height_slider"], {
            start: [sigmajs_default_settings["height_original"]],
            connect: true,
            range: {
                'min': sigmajs_default_settings["height_original"],
                'max': sigmajs_default_settings["height_max"],
            },
            step: sigmajs_default_settings["height_step"],
            pips: {
                mode: 'count',
                values: sigmajs_default_settings["height_number_options"],
                density: 1
            }
        });

        if (sigmajs_bn_settings["nodes_slider"].noUiSlider) {
            sigmajs_bn_settings["nodes_slider"].noUiSlider.destroy();
        }

        noUiSlider.create(sigmajs_bn_settings["nodes_slider"], {
            start: [sigmajs_default_settings["maxNodeSize"]],
            connect: true,
            format: wNumb({
                decimals: 2
            }),
            range: {
                'min': sigmajs_default_settings["minNodeSize"],
                'max': sigmajs_default_settings["maxNodeSize"],
            },
            step: sigmajs_default_settings["nodeSizeStep"],
            pips: {
                mode: 'count',
                values: 5,
                density: 1,
                format: wNumb({
                    decimals: 1,
                }),
            }
        });

        if (sigmajs_bn_settings["edges_slider"].noUiSlider) {
            sigmajs_bn_settings["edges_slider"].noUiSlider.destroy();
        }
        noUiSlider.create(sigmajs_bn_settings["edges_slider"], {
            start: [sigmajs_default_settings["maxEdgeSize"]],
            connect: true,
            format: wNumb({
                decimals: 2
            }),
            range: {
                'min': sigmajs_default_settings["minEdgeSize"],
                'max': sigmajs_default_settings["maxEdgeSize"],
            },
            step: sigmajs_default_settings["edgeSizeStep"],
            pips: {
                mode: 'count',
                values: 5,
                density: 1,
                format: wNumb({
                    decimals: 1,
                }),
            }
        });

        if ($("#model_name").val() !== "ml_probabilistic_clustering") {
            if (sigmajs_bn_settings["filter_edges_weight_slider"].noUiSlider) {
                sigmajs_bn_settings["filter_edges_weight_slider"].noUiSlider.destroy();
            }
            if (sigmajs_default_settings["minWeight"] !== "" && sigmajs_default_settings["minWeight"] !== sigmajs_default_settings["maxWeight"]) {
                $("#filter-edges-by-weight-section").removeClass("hide-element").show();

                noUiSlider.create(sigmajs_bn_settings["filter_edges_weight_slider"], {
                    start: [sigmajs_default_settings["minWeight"], sigmajs_default_settings["maxWeight"]],
                    connect: true,
                    format: wNumb({
                        decimals: 3
                    }),
                    range: {
                        'min': sigmajs_default_settings["minWeight"],
                        'max': sigmajs_default_settings["maxWeight"],
                    },
                    step: 0.001,
                    pips: {
                        mode: 'count',
                        values: 3,
                        density: 1,
                        format: wNumb({
                            decimals: 3,
                        }),
                    }
                });

                set_input_left_for_slider("slider-filter-edges-by-weight", sigmajs_bn_settings["filter_edges_weight_slider"], sigmajs_default_settings["minWeight"], function () {
                });
                set_input_rigth_for_slider("slider-filter-edges-by-weight", sigmajs_bn_settings["filter_edges_weight_slider"], sigmajs_default_settings["maxWeight"], function () {
                });
            } else {
                $("#filter-edges-by-weight-section").hide();
            }
        }


        sigmajs_bn_settings["width_old"] = sigmajs_default_settings["widht_original"];
        sigmajs_bn_settings["height_old"] = sigmajs_default_settings["height_old"];
    }


    function fill_node_input(input_select_name, nodes) {
        var select_input = $("#" + input_select_name);
        select_input.selectize()[0].selectize.destroy();
        select_input.empty();
        var items = nodes.map(function (x) {
            return {item: x};
        });
        if (input_select_name === "select-start-nodes-bn") {
            select_input.selectize({
                maxOptions: nodes.length,
                labelField: "item",
                valueField: "item",
                searchField: "item",
                options: items,
            });
        } else if (input_select_name === "select-end-nodes-bn") {
            select_input.selectize({
                maxOptions: nodes.length,
                labelField: "item",
                valueField: "item",
                searchField: "item",
                options: items,
            });
        } else {
            select_input.selectize({
                maxOptions: nodes.length,
                labelField: "item",
                valueField: "item",
                searchField: "item",
                options: items,
            });
        }
    }


    function init_sigmajs_plugins_after_creation(sigmajs_default_settings, nodes, additional_discrete_features) {
        //To have a copy of the original colors in case some plugin modify them:
        save_original_color_nodes_edges(sigmajs_bn, sigmajs_default_settings);

        if ($("#model_name").val() === "ml_probabilistic_clustering") {
            initialize_selectize_structures(sigmajs_bn_settings["structures_info"]);
        }

        print_current_num_nodes_edges();

        //Drag nodes:
        var dragListener = sigma.plugins.dragNodes(sigmajs_bn, sigmajs_bn.renderers[0]);

        $("#checkbox-show-hide-labels").prop('checked', sigmajs_default_settings["drawLabels"]);

        //Initial behaviour of "on_click_node" is defined by the html "checked" parameters in the checkboxes
        on_change_click_node_options(sigmajs_default_settings);
        on_change_click_edge_options(sigmajs_default_settings);

        //Click outside of a node:
        sigmajs_bn.bind('clickStage', function (e) {
            $("#node-parameters-plot-copy").remove();
        });

        //Double click stage:
        sigmajs_bn.bind('doubleClickStage', function (e) {
            sigmajs_bn.settings('doubleClickEnabled', false);
            reset_nodes_edges_color(sigmajs_bn);
            //resetCameraSigmajs(sigmajs_bn);
            sigmajs_bn.refresh();
        });

        //Search nodes input:
        fill_node_input("select-node-bn", nodes);
        fill_node_input("select-multi-nodes-bn", nodes);
        if ($("#model_name").val() !== "ml_probabilistic_clustering") {
            fill_node_input("select-start-nodes-bn", nodes);
            fill_node_input("select-observed-nodes-bn", nodes);
            fill_node_input("select-end-nodes-bn", nodes);
        }
        var select_node_input = $("#select-node-bn");
        select_node_input.on("change", function (e) {
            var node_chosen_id = $(this).val();
            if (node_chosen_id !== "" && node_chosen_id !== undefined) {
                focusOnNodeSigmajs(sigmajs_bn, node_chosen_id);
                focus_single_node(node_chosen_id);
            }
        });

        // Search groups input:
        set_additional_discrete_features(additional_discrete_features);

        sigmajs_bn_settings["width_slider"].noUiSlider.on("update", function (values, handle) {
            var width_now = parseFloat(values[handle]);
            var width_next = width_now + sigmajs_default_settings["width_step"];

            sigmajs_bn.graph.nodes().forEach(function (node) {
                if (width_now > sigmajs_bn_settings["width_old"]) {
                    node.x /= width_now;
                } else if (width_now < sigmajs_bn_settings["width_old"]) {
                    node.x *= width_next;
                }
            });
            sigmajs_bn_settings["width_old"] = width_now;
            sigmajs_bn.refresh();
        });

        sigmajs_bn_settings["height_slider"].noUiSlider.on("update", function (values, handle) {
            var height_now = parseFloat(values[handle]);
            var height_next = height_now + sigmajs_default_settings["height_step"];

            sigmajs_bn.graph.nodes().forEach(function (node) {
                if (height_now > sigmajs_bn_settings["height_old"]) {
                    node.y /= height_now;
                } else if (height_now < sigmajs_bn_settings["height_old"]) {
                    node.y *= height_next;
                }
            });
            sigmajs_bn_settings["height_old"] = height_now;
            sigmajs_bn.refresh();
        });

        sigmajs_bn_settings["nodes_slider"].noUiSlider.on("update", function (values, handle) {
            var nodes_size = parseFloat(values[handle]);
            sigmajs_bn.settings('maxNodeSize', nodes_size);
            sigmajs_bn.refresh();

        });
        sigmajs_bn_settings["edges_slider"].noUiSlider.on("update", function (values, handle) {
            var edges_size = parseFloat(values[handle]);
            sigmajs_bn.settings('maxEdgeSize', edges_size);
            sigmajs_bn.refresh();
        });

        $("#checkbox-scale-nodes-dependent-markov-blanket").on("change", function (e) {
            var scale_nodes_dependent = $(this).is(":checked");
            if (scale_nodes_dependent) {
                var data_client_json = {
                    "model_name": $("#model_name").val()
                };
                data_client_json = JSON.stringify(data_client_json);
                var data_send = {
                    "csrfmiddlewaretoken": csrf_token,
                    "data_client_json": data_client_json,
                };
                $.ajax({
                    type: "POST",
                    url: "/morpho/ml_bn_nodes_sizes_dependent_markov_blanket/",
                    data: data_send,
                    dataType: 'json',
                    success: function (data) {
                        if (!$.isEmptyObject(data["nodes_sizes"])) {
                            sigmajs_bn.graph.nodes().forEach(function (n) {
                                n.size = data["nodes_sizes"][n.id];
                            });
                            sigmajs_bn.settings('minNodeSize', data["min_node_size"]);
                            sigmajs_bn.settings('maxNodeSize', data["max_node_size"]);
                            sigmajs_bn.refresh();
                        }
                    },
                    error: function () {
                    }
                });
            } else {
                sigmajs_bn.graph.nodes().forEach(function (n) {
                    sigmajs_bn.settings('minNodeSize', sigmajs_default_settings["minNodeSize"]);
                    sigmajs_bn.settings('maxNodeSize', sigmajs_default_settings["maxNodeSize"]);
                    n.size = sigmajs_default_settings["maxNodeSize"];
                });
                sigmajs_bn.refresh();
            }
        });

        $("#checkbox-scale-nodes-dependent-neighbors").on("change", function (e) {
            var scale_nodes_dependent = $(this).is(":checked");
            if (scale_nodes_dependent) {
                var data_client_json = {
                    "model_name": $("#model_name").val()
                };
                data_client_json = JSON.stringify(data_client_json);
                var data_send = {
                    "csrfmiddlewaretoken": csrf_token,
                    "data_client_json": data_client_json,
                };
                $.ajax({
                    type: "POST",
                    url: "/morpho/ml_bn_nodes_sizes_dependent_neighbors/",
                    data: data_send,
                    dataType: 'json',
                    success: function (data) {
                        if (!$.isEmptyObject(data["nodes_sizes"])) {
                            sigmajs_bn.graph.nodes().forEach(function (n) {
                                n.size = data["nodes_sizes"][n.id];
                            });
                            sigmajs_bn.settings('minNodeSize', data["min_node_size"]);
                            sigmajs_bn.settings('maxNodeSize', data["max_node_size"]);
                            sigmajs_bn.refresh();
                        }
                    },
                    error: function () {
                    }
                });
            } else {
                sigmajs_bn.graph.nodes().forEach(function (n) {
                    sigmajs_bn.settings('minNodeSize', sigmajs_default_settings["minNodeSize"]);
                    sigmajs_bn.settings('maxNodeSize', sigmajs_default_settings["maxNodeSize"]);
                    n.size = sigmajs_default_settings["maxNodeSize"];
                });
                sigmajs_bn.refresh();
            }
        });

        $("#checkbox-scale-edges-dependent-weights").on("change", function (e) {
            var scale_edges_dependent = $(this).is(":checked");
            if (scale_edges_dependent) {
                var data_client_json = {
                    "model_name": $("#model_name").val()
                };
                data_client_json = JSON.stringify(data_client_json);
                var data_send = {
                    "csrfmiddlewaretoken": csrf_token,
                    "data_client_json": data_client_json,
                };
                $.ajax({
                    type: "POST",
                    url: "/morpho/ml_bn_edges_size_dependent_weights/",
                    data: data_send,
                    dataType: 'json',
                    success: function (data) {
                        if (!$.isEmptyObject(data["edges_sizes"])) {
                            sigmajs_bn.graph.edges().forEach(function (edge) {
                                edge.size = data["edges_sizes"][edge.source + data["edges_sep_char"] + edge.target];
                            });
                            sigmajs_bn.settings('minEdgeSize', data["min_edge_size"]);
                            sigmajs_bn.settings('maxEdgeSize', data["max_edge_size"]);
                            sigmajs_bn.refresh();
                        }
                    },
                    error: function (e) {
                    }
                });
            } else {
                sigmajs_bn.graph.edges().forEach(function (edge) {
                    sigmajs_bn.settings('minEdgeSize', sigmajs_default_settings["minEdgeSize"]);
                    sigmajs_bn.settings('maxEdgeSize', sigmajs_default_settings["maxEdgeSize"]);
                    edge.size = sigmajs_default_settings["maxEdgeSize"];
                });
                sigmajs_bn.refresh();
            }
        });

        $("#checkbox-highlight-important-nodes").on("change", function () {
            var highlight_important_nodes = $(this).is(":checked");
            var num_neighbors = $("#n-neighbors-important").val();
            var include_neighbors = $("#checkbox-include-important-neighbors").is(":checked");
            var propagate_colors_to_hubs_neighbors = true;
            var highlight_algorithm_selected = $("#select-highlight-important-nodes-algorithm").val();
            if (highlight_important_nodes) {
                if (highlight_algorithm_selected === "degrees") {
                    get_important_nodes_degrees(highlight_algorithm_selected, sigmajs_default_settings, num_neighbors, include_neighbors, propagate_colors_to_hubs_neighbors);
                } else if (highlight_algorithm_selected === "betweenness-centrality") {
                    get_important_nodes_betweenness(highlight_algorithm_selected, sigmajs_default_settings);
                }
            } else {
                sigmajs_bn.graph.nodes().forEach(function (n) {
                    change_node_edge_color(n, n.originalColor, false);
                    n.selected_important = false;

                    sigmajs_bn.settings('maxNodeSize', sigmajs_default_settings["maxNodeSize"]);
                    n.size = sigmajs_default_settings["maxNodeSize"];
                });
                sigmajs_bn_settings["nodes_slider"].noUiSlider.updateOptions({
                    start: [sigmajs_bn.settings('maxNodeSize')],
                    range: {
                        'min': sigmajs_bn.settings('minNodeSize'),
                        'max': parseFloat(sigmajs_bn.settings('maxNodeSize').toFixed(1))
                    }
                });
                sigmajs_bn.refresh();
            }

        });

        $("#checkbox-highlight-communities").on("change", function () {
            var highlight_communities = $(this).is(":checked");
            var highlight_algorithm_selected = $("#select-highlight-communities-algorithm").val();
            if (highlight_communities) {
                if (highlight_algorithm_selected === "louvain") {
                    var notify_loading = print_notify_message(permanent = true, type = "info",
                        icon = 'fa fa-spinner fa-spin icon-margin-right', title = "Loading louvain",
                        message = "Loading louvain algorithm", delay = 0,
                        additional_message = "", position = "bottom-right");
                    var data_client_json = {
                        "selection_option": highlight_algorithm_selected,
                        "model_name": $("#model_name").val()
                    };
                    data_client_json = JSON.stringify(data_client_json);

                    var data_send = {
                        "csrfmiddlewaretoken": csrf_token,
                        "data_client_json": data_client_json
                    };
                    $.ajax({
                        type: "POST",
                        url: "/morpho/ml_bn_highlight_communities/",
                        data: data_send,
                        dataType: 'json',
                        success: function (data) {
                            if (!$.isEmptyObject(data["communities"]) && !$.isEmptyObject(data["additional_parameters"])) {
                                for (var node_id in data["communities"]) {
                                    var node = data["communities"][node_id];
                                    var node_sigmajs = sigmajs_bn.graph.getNode(node_id);
                                    change_node_edge_color(node_sigmajs, node.color, true);
                                    node_sigmajs.selected_important = true;
                                }
                                set_additional_discrete_features(data["additional_parameters"]);
                                sigmajs_bn.refresh();
                            }
                        }, error: function () {
                        },
                        complete: function () {
                            notify_loading.close();
                        }
                    });
                }
            } else {
                var data_client_json = {
                    "selection_option": highlight_algorithm_selected,
                    "model_name": $("#model_name").val()
                };
                data_client_json = JSON.stringify(data_client_json);
                var data_send = {
                    "csrfmiddlewaretoken": csrf_token,
                    "data_client_json": data_client_json
                };
                $.ajax({
                    type: "POST",
                    url: "/morpho/ml_bn_restore_highlight_communities/",
                    data: data_send,
                    dataType: 'json',
                    success: function (data) {
                        if (!$.isEmptyObject(data["additional_parameters"])) {
                            set_additional_discrete_features(data["additional_parameters"]);
                        } else {
                            set_additional_discrete_features();
                        }
                        sigmajs_bn.refresh();
                    }, error: function () {
                    }
                });

                sigmajs_bn.graph.nodes().forEach(function (n) {
                    if (n.evidence_set === true) {
                        change_node_edge_color(n, sigmajs_bn_settings["evidence_set_color"], false);
                        n.selected_important = true;
                    } else {
                        change_node_edge_color(n, sigmajs_bn_settings["original_color"], true);
                        n.selected_important = false;
                    }
                    n.label = n.id;
                    sigmajs_bn.settings('maxNodeSize', sigmajs_default_settings["maxNodeSize"]);
                    n.size = sigmajs_default_settings["maxNodeSize"];
                });
                sigmajs_bn.graph.edges().forEach(function (e) {
                    change_node_edge_color(e, e.originalColor, false);
                    e.selected = false;
                });
                sigmajs_bn_settings["nodes_slider"].noUiSlider.updateOptions({
                    start: [sigmajs_bn.settings('maxNodeSize')],
                    range: {
                        'min': sigmajs_bn.settings('minNodeSize'),
                        'max': parseFloat(sigmajs_bn.settings('maxNodeSize').toFixed(1))
                    }
                });
                sigmajs_bn.refresh();
            }
        });

        $("#select-highlight-important-nodes-algorithm").on("change", function (e) {
            var highlight_algorithm_selected = $("#select-highlight-important-nodes-algorithm").val();
            var n_neighbors_important_form = $("#n-neighbors-important-form");
            var include_important_neighbors_form = $("#include-important-neighbors-form");

            if (highlight_algorithm_selected === "degrees") {
                n_neighbors_important_form.removeClass("hide-element");
                include_important_neighbors_form.removeClass("hide-element");
            } else {
                n_neighbors_important_form.addClass("hide-element");
                include_important_neighbors_form.addClass("hide-element");
            }
        });

        //Prevent on click node options to close the dropdown menu:
        $(".buttons-options-draw-bn").on('click', '.dropdown-menu ', function (e) {
            e.stopPropagation();
        });

        $("#select-highlight-important-nodes-algorithm").on('click', function (e) {
            e.stopPropagation();
        });

        $("#select-highlight-communities-algorithm").on('click', function (e) {
            e.stopPropagation();
        });

        $(".checkbox-on-click-node-option").on("change", function (e) {
            on_change_click_node_options(sigmajs_default_settings);
        });


        $(".checkbox-on-click-edge-option").on("change", function (e) {
            on_change_click_edge_options(sigmajs_default_settings);
        });

    }

    function set_additional_discrete_features(additional_discrete_features) {
        if (additional_discrete_features) {
            $("#bn-options-on-click-group").removeClass("hide-element");
            $("#bn-options-on-click-group").show();
            $("#select-category-graph").hide();

            $("#select-group-graph").removeClass("hide-element");
            $("#select-group-graph").show();
            var select_groups_input = $("#select-group-bn");
            select_groups_input.selectize()[0].selectize.destroy();
            select_groups_input.empty();
            var items = additional_discrete_features.map(function (x) {
                return {item: x};
            });
            var current_item = additional_discrete_features[additional_discrete_features.length - 1];
            items.push({"item": "No group"});
            select_groups_input.selectize({
                maxItems: 1,
                maxOptions: additional_discrete_features.length + 1,
                labelField: "item",
                valueField: "item",
                searchField: "item",
                options: items,
            });

            if ($("#model_name").val() === "ml_probabilistic_clustering" && $("#datatable_available").val() === "True") {
                init_update_selectize_datatable_cols_groups();
            }

            select_groups_input.on("change", function (e) {
                var group_chosen_id = $(this).val();
                if (group_chosen_id == "No group") {
                    $("#button-set-evidence-group").addClass("hide-element");
                    $("#button-clear-evidence-group").addClass("hide-element");
                    sigmajs_bn_settings["active_group"] = "";
                    sigmajs_bn_settings["active_category"] = "";
                }
                get_and_render_all_nodes_by_group(sigmajs_bn, group_chosen_id);
                get_category_values_in_group(group_chosen_id);
            });

            var selectize_groups_input = select_groups_input[0].selectize;
            selectize_groups_input.addItem(current_item);
        } else {
            $("#bn-options-on-click-group").hide();
            $("#select-group-graph").hide();
        }
    }

    function on_change_click_node_options(sigmajs_default_settings) {
        var show_markov_blanket = $("#checkbox-on-click-markov-blanket").is(":checked");
        var show_neighbors = $("#checkbox-on-click-neighbors").is(":checked");
        var show_parents = $("#checkbox-on-click-parents").is(":checked");
        var show_children = $("#checkbox-on-click-children").is(":checked");
        var show_parameters = $("#checkbox-on-click-parameters").is(":checked");
        var show_connections_info = $("#checkbox-on-click-connections-info").is(":checked");
        var show_same_category = $("#checkbox-on-click-same-group").is(":checked");
        on_click_node_options(sigmajs_default_settings, show_markov_blanket, show_neighbors, show_parents, show_children, show_parameters, show_same_category, show_connections_info)
    }

    function on_change_click_edge_options(sigmajs_default_settings) {
        var show_edge_info = $("#checkbox-on-click-edge-info").is(":checked");
        on_click_edge_options(sigmajs_default_settings, show_edge_info)
    }

    function on_click_node_options(sigmajs_default_settings, show_markov_blanket, show_neighbors, show_parents, show_children, show_parameters, show_same_category, show_connections_info) {
        sigmajs_bn.unbind('clickNode');
        sigmajs_bn.bind('clickNode', function (e) {
            var node_chosen_id = e.data.node.id;
            if ($("#model_name").val() !== "ml_probabilistic_clustering") {
                reset_nodes_color(sigmajs_bn);
            }
            copyToClipboard(node_chosen_id);

            if (show_parameters) {
                get_and_render_node_parameters(sigmajs_bn, node_chosen_id, e.data.captor);
            }
            if (show_same_category) {
                get_and_render_nodes_same_category(sigmajs_bn, node_chosen_id, sigmajs_default_settings["highlightNodeColor"]);
            }
            if (show_markov_blanket) {
                var url = "/morpho/ml_bn_get_markov_blanket/";
                var selection = "markov_blanket";
                get_and_render_x_nodes(sigmajs_bn, node_chosen_id, sigmajs_default_settings["highlightNodeColor"], url, selection);
            }
            if (show_neighbors) {
                var url = "/morpho/ml_bn_get_neighbors/";
                var selection = "neighbors";
                get_and_render_x_nodes(sigmajs_bn, node_chosen_id, sigmajs_default_settings["highlightNodeColor"], url, selection);
            }
            if (show_parents) {
                var url = "/morpho/ml_bn_get_parents/";
                var selection = "parents";
                get_and_render_x_nodes(sigmajs_bn, node_chosen_id, sigmajs_default_settings["highlightNodeColor"], url, selection);
            }
            if (show_children) {
                var url = "/morpho/ml_bn_get_children/";
                var selection = "children";
                get_and_render_x_nodes(sigmajs_bn, node_chosen_id, sigmajs_default_settings["highlightNodeColor"], url, selection);
            }
            if (show_connections_info) {
                get_and_render_node_connections_info(sigmajs_bn, node_chosen_id, sigmajs_default_settings["highlightNodeColor"]);
            }
        });
    }

    function on_click_edge_options(sigmajs_default_settings, show_edge_info) {
        //sigmajs_bn_settings["drawing_engine"] = "canvas";

        sigmajs_bn.unbind('clickEdge');
        sigmajs_bn.bind('clickEdge', function (e) {
            var source_node = e.data.edge.source;
            var target_node = e.data.edge.target;
            reset_nodes_edges_color(sigmajs_bn);

            if (show_edge_info) {
                get_and_render_edge_info(sigmajs_bn, source_node, target_node);
            }
        });

    }

    function reload_sigmajs_graph(sigmajs_instance) {
        // var layout_name = undefined;
        // var additional_params = undefined;
        // change_layout_sigmajs(layout_name, additional_params);
        $("#continue-button-upload-bn").trigger("click");
    }

    function focusOnNodeSigmajs(sigmajs_instance, node_id) {
        var node = sigmajs_instance.graph.getNode(node_id);

        //resetCameraSigmajs(sigmajs_bn);


        sigma.misc.animation.camera(
            sigmajs_instance.camera,
            {
                x: node[sigmajs_instance.camera.readPrefix + 'x'],
                y: node[sigmajs_instance.camera.readPrefix + 'y'],
                ratio: 0.15
            },
            {
                duration: 500
            }
        );
    }

    function resetCameraSigmajs(sigmajs_instance) {
        sigma.misc.animation.camera(
            sigmajs_instance.camera,
            {
                x: 0,
                y: 0,
                angle: 0,
                ratio: 1.1
            },
            {
                duration: 500
            }
        );
    }

    function save_original_color_nodes_edges(sigmajs_instance, sigmajs_default_settings) {
        sigmajs_instance.graph.nodes().forEach(function (n) {
            n.originalColor = n.color;
        });
        sigmajs_instance.graph.edges().forEach(function (e) {
            e.originalColor = e.color;
        });

        sigmajs_bn_settings["color_hidden"] = sigmajs_default_settings["color_hidden"];
        sigmajs_bn_settings["color_common_edges"] = sigmajs_default_settings["color_common_edges"];
        sigmajs_bn_settings["color_structure_1"] = sigmajs_default_settings["color_structure_1"];
        sigmajs_bn_settings["color_structure_2"] = sigmajs_default_settings["color_structure_2"];
        sigmajs_bn_settings["highlight_node_color"] = sigmajs_default_settings["highlightNodeColor"];
        sigmajs_bn_settings["structures_info"] = sigmajs_default_settings["structures_info"];
    }

    function hide_node_edge(x) {
        var hide = $("#checkbox-on-click-group-hide-others").is(":checked");
        if (hide) {
            x.hidden = true;
        } else {
            x.color = sigmajs_bn_settings["color_hidden"];
            x.hidden = false;
        }
        x.selected = false;
    }

    function change_node_edge_color(x, color, important) {
        x.hidden = false;

        x.color = color;
        if (important) {
            x.color_important = color;
        }
    }

    function reset_node_color(n) {
        if (!n.selected_important) {
            change_node_edge_color(n, sigmajs_bn_settings["original_color"], false);
            n.selected = false;
        } else {
            if (n.evidence_set) {
                change_node_edge_color(n, sigmajs_bn_settings["evidence_set_color"], false);
            } else {
                change_node_edge_color(n, n.color_important, false);
            }
            n.selected = false;
        }
    }

    function reset_nodes_color(sigmajs_instance) {
        sigmajs_instance.graph.nodes().forEach(function (n) {
            reset_node_color(n);
        });

        sigmajs_instance.refresh();
    }

    function reset_edges_color(sigmajs_instance) {
        sigmajs_instance.graph.edges().forEach(function (e) {
            if (!e.structure_selected || e.selected) {
                change_node_edge_color(e, e.originalColor, false);
                e.selected = false;
            }
        });

        sigmajs_instance.refresh();
    }

    function reset_nodes_edges_color(sigmajs_instance) {
        reset_nodes_color(sigmajs_instance);

        if (sigmajs_bn_settings["active_category"] !== "") {
            $("#select-category-bn")[0].selectize.addItem("All categories");
        }
        sigmajs_bn_settings["active_category"] = "";

        print_current_num_nodes_edges();

        if ($("#model_name").val() === "ml_probabilistic_clustering") {
            reset_nodes_edges_color_prob_clustering();
        } else {
            reset_edges_color(sigmajs_instance);
        }
    }

    function render_x_nodes(nodes_selection, sigmajs_instance, selection, nodeId, highlight_node_color) {
        var toKeep = {};
        for (var i = 0; i < nodes_selection.length; i++) {
            var node_id = nodes_selection[i];
            toKeep[node_id] = sigmajs_instance.graph.getNode(node_id)
        }

        sigmajs_instance.graph.nodes().forEach(function (n) {
            var valid = true;

            if (sigmajs_bn_settings["active_category"] !== "" && !n.selected) {
                valid = false;
            }

            if (valid) {
                if (toKeep[n.id]) {
                    if (!n.selected_important) {
                        change_node_edge_color(n, n.originalColor, false);
                    } else {
                        change_node_edge_color(n, n.color_important, false);
                    }
                    n.selected = true;
                } else {
                    hide_node_edge(n);
                }
            } else {
                hide_node_edge(n)
            }
        });

        sigmajs_instance.graph.edges().forEach(function (e) {
            var source = e.source;
            var target = e.target;
            var valid_edge = true;
            var highlight_edge = false;

            if (selection === "parents" && target !== nodeId) {
                valid_edge = false;
            }
            if (selection === "children" && source !== nodeId) {
                valid_edge = false;
            }
            if (sigmajs_bn_settings["active_category"] !== "" && !e.selected) {
                valid_edge = false;
            }

            if (valid_edge) {
                if (toKeep[source] && toKeep[target]) {
                    change_node_edge_color(e, e.color, false);
                    e.selected = true;
                    highlight_edge = true;
                }
            }
            if (!highlight_edge) {
                hide_node_edge(e);
            }
        });

        sigmajs_instance.refresh();

        set_node_color(sigmajs_instance, nodeId, highlight_node_color);
    }

    function get_and_render_x_nodes(sigmajs_instance, nodeId, highlight_node_color, url, selection) {
        var data_client_json = {
            "node_id": nodeId,
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json);
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: url,
            data: data_send,
            dataType: 'json',
            success: function (data) {
                var nodes_selection = data["nodes_selection"];
                if (!nodes_selection.includes(nodeId)) {
                    nodes_selection.push(nodeId);
                }
                render_x_nodes(nodes_selection, sigmajs_instance, selection, nodeId, highlight_node_color);
            },
            error: function () {
            }
        });
    }

    function get_and_render_all_nodes_by_group(sigmajs_instance, group_id) {
        var data_client_json = {
            "group_id": group_id,
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json);
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_get_info_nodes_by_group/",
            data: data_send,
            dataType: 'json',
            success: function (data) {
                if (!$.isEmptyObject(data["info_nodes"])) {
                    sigmajs_bn_settings["active_group"] = group_id;
                    sigmajs_instance.graph.nodes().forEach(function (n) {
                        if (data["info_nodes"][n.id]) {
                            change_node_edge_color(n, data["info_nodes"][n.id]["color"], true);
                            n.selected_important = true;
                            n.label = n.id + " | (" + data["info_nodes"][n.id]["category"] + ")";
                        } else if (!n.evidence_set) {
                            n.originalColor = sigmajs_bn_settings["original_color"];
                            change_node_edge_color(n, sigmajs_bn_settings["original_color"], true);
                            n.selected_important = false;
                            n.label = n.id;
                            n.selected = false;
                        }
                    });
                } else {
                    sigmajs_bn_settings["active_group"] = "";
                    sigmajs_instance.graph.nodes().forEach(function (n) {
                        if (!n.evidence_set) {
                            n.originalColor = sigmajs_bn_settings["original_color"];
                            change_node_edge_color(n, sigmajs_bn_settings["original_color"], true);
                            n.selected_important = false;
                            n.label = n.id;
                            n.selected = false;
                        }
                    });
                }

                sigmajs_instance.refresh();
            },
            error: function () {
            }
        });
    }

    function get_all_nodes_in_category(sigmajs_instance, category_id, option, notify_loading) {
        var notify_loading = print_notify_message(permanent = true, type = "info",
            icon = 'fa fa-spinner fa-spin icon-margin-right', title = "Filtering",
            message = "Loading nodes and edges.", delay = 0,
            additional_message = "", position = "bottom-right");

        var show_neighbors = $("#checkbox-on-click-group-show-neighbors").is(":checked");
        var data_client_json = {
            "group_id": sigmajs_bn_settings["active_group"],
            "category_id": category_id,
            "structure_id": sigmajs_bn_settings["active_structure"],
            "show_neighbors": show_neighbors,
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json);
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_get_nodes_in_category/",
            data: data_send,
            dataType: 'json',
            success: function (data) {
                notify_loading.close();
                if (!data["error"]) {
                    sigmajs_bn_settings["active_category"] = category_id;

                    if (option === "render_nodes_in_category") {
                        reset_nodes_color(sigmajs_bn);
                        render_nodes_in_category(sigmajs_instance, data["nodes_in_category"], data["color_category"]);
                        if (show_neighbors) {
                            render_nodes_in_category(sigmajs_instance, data["neighbors"], undefined);
                            var nodes_to_highlight = data["nodes_in_category"].concat(data["neighbors"]);
                            get_edges_between_nodes(nodes_to_highlight);
                        } else {
                            get_edges_between_nodes(data["nodes_in_category"]);
                        }
                    } else if (option === "clear_nodes_evidences") {
                        $("#node-parameters-plot-copy").remove();
                        clear_nodes_evidences(data["nodes_in_category"], {}, true);
                    } else if (option === "set_nodes_evidences") {
                        var evidence_group_value = $("#evidence-group-value");
                        var new_evidence_value = evidence_group_value.val();
                        $("#node-parameters-plot-copy").remove();
                        set_nodes_evidences(data["nodes_in_category"], new_evidence_value, {}, true, callback_function = function () {
                            notify_loading.close();
                        });
                    }
                }
            },
            error: function (e) {
                notify_loading.close();
            }
        });
    }

    function get_and_render_nodes_same_category(sigmajs_instance, nodeId, highlight_node_color) {
        if (sigmajs_bn_settings["active_group"] === "") {
            var notify_error = print_notify_message(permanent = false, type = "danger",
                icon = 'fa fa-warning icon-margin-right', title = "Error showing the category (" + nodeId + "): ",
                message = "First select a group and category at the left side of the screen", delay = 400,
                additional_message = "", position = "bottom-right");
        } else {
            var data_client_json = {
                "group_id": sigmajs_bn_settings["active_group"],
                "node_id": nodeId,
                "model_name": $("#model_name").val()
            };
            data_client_json = JSON.stringify(data_client_json)
            var data_send = {
                "csrfmiddlewaretoken": csrf_token,
                "data_client_json": data_client_json,
            };

            $.ajax({
                type: "POST",
                url: "/morpho/ml_bn_get_nodes_same_category/",
                data: data_send,
                dataType: 'json',
                success: function (data) {
                    render_nodes_in_category(sigmajs_instance, data["nodes_in_category"], data["color_category"]);
                    get_edges_between_nodes(data["nodes_in_category"]);
                    set_node_color(sigmajs_instance, nodeId, highlight_node_color);
                },
                error: function () {
                }
            });
        }
    }

    function render_nodes_in_category(sigmajs_instance, nodes_in_category, color_category) {
        if (nodes_in_category !== undefined) {
            var toKeep = {};
            var num_nodes_selected = 0;
            for (var i = 0; i < nodes_in_category.length; i++) {
                var node_id = nodes_in_category[i];
                toKeep[node_id] = sigmajs_instance.graph.getNode(node_id)
            }

            if (color_category === undefined) {
                color_category = sigmajs_bn_settings["original_color"];
            }

            sigmajs_instance.graph.nodes().forEach(function (n) {
                if (toKeep[n.id]) {
                    if (n.evidence_set) {
                        change_node_edge_color(n, sigmajs_bn_settings["evidence_set_color"], false);
                    } else {
                        change_node_edge_color(n, color_category, false);
                    }
                    n.selected = true;
                    n.selected_important = true;
                    num_nodes_selected++;
                } else {
                    if (!n.selected) {
                        hide_node_edge(n)
                    } else {
                        num_nodes_selected++;
                    }
                }
            });

            sigmajs_instance.graph.edges().forEach(function (e) {
                if (!e.selected) {
                    hide_node_edge(e);
                }
            });
        } else {
            reset_nodes_edges_color(sigmajs_instance);
        }

        $("#network-stats-num-nodes-selected").text(num_nodes_selected);

        sigmajs_instance.refresh();
    }

    function set_node_color(sigmajs_instance, nodeId, rgb_color) {
        var node = sigmajs_instance.graph.getNode(nodeId);
        change_node_edge_color(node, rgb_color, false);
        node.selected = true;

        sigmajs_instance.refresh();
    }

    function render_continuous_params(placeholder_node_plot, parameters_plot, node_parameters_selection, evidence_value) {
        placeholder_node_plot.append(parameters_plot);
        var chart = placeholder_node_plot.find(".plotly-graph-div");
        var plot_maximum = chart[0].data[0].x[chart[0].data[0].x.length - 1];
        var plot_minimum = chart[0].data[0].x[0];

        node_parameters_selection.attr({
            "min": chart[0].data[0].x[0],
            "max": chart[0].data[0].x[chart[0].data[0].x.length - 1],
            "step": ((plot_maximum - plot_minimum) / 100).toFixed(3)
        });
        node_parameters_selection.val(evidence_value);

        placeholder_node_plot.find(".plotly-graph-div").on('plotly_click', function (data, points) {
            var new_evidence_value = points.points[0].x;
            node_parameters_selection.val(new_evidence_value);
        });
    }

    function show_waiting_inference(node_probability_continuous_parameters, node_probability_discrete_parameters, evidence_buttons, waiting_div) {
        node_probability_continuous_parameters.hide();
        node_probability_discrete_parameters.hide();
        evidence_buttons.hide();
        waiting_div.find('#text-waiting-param-node').text("Running inference");
        waiting_div.show();
    }

    function get_and_render_node_parameters(sigmajs_instance, nodeId, node_data_captor) {
        var node_sigmajs = sigmajs_bn.graph.getNode(nodeId);
        var inference_mode = $("#bn-evidence-mode").is(":checked");
        var placeholder_node_params = $("#node-parameters-plot-root");
        var bayesian_network_graph = $("#bayesian-network-graph");
        $("#node-parameters-plot-copy").remove();
        var placeholder_node_params_copy = placeholder_node_params.clone();
        placeholder_node_params_copy.attr("id", "node-parameters-plot-copy");
        bayesian_network_graph.append(placeholder_node_params_copy);
        placeholder_node_params_copy.removeClass("hide-element").show();

        bayesian_network_graph.css({position: 'relative'});
        var parameters_height = (bayesian_network_graph.height() / 2) + node_data_captor.y - (placeholder_node_params_copy.height() + 60);
        var parameters_width = (bayesian_network_graph.width() / 2) + node_data_captor.x - (placeholder_node_params_copy.width() / 2);
        placeholder_node_params_copy.css({top: parameters_height, left: parameters_width, position: 'absolute'});


        var data_client_json = {
            "node_id": nodeId,
            "inference_mode": inference_mode,
            "structure_id": $("#selectize-bn-structure").val(),
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json);
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_get_node_parameters/",
            data: data_send,
            dataType: 'json',
            success: function (data) {
                if (data["error"]) {
                    placeholder_node_params_copy.remove();
                    var message_error = "No parameters available for this node. <br> Learn them in the section above or upload them.";
                    var notify_error = print_notify_message(permanent = false, type = "danger",
                        icon = 'fa fa-align-left icon-margin-right', title = "Parameters (" + nodeId + "): ",
                        message = message_error, delay = 150,
                        additional_message = "", position = "bottom-right");
                } else {
                    var data_type = data["data_type"];
                    var node_probability_discrete_parameters = $('#node-parameters-plot-copy #node-probability-discrete-parameters');
                    var node_probability_continuous_parameters = $('#node-parameters-plot-copy #node-probability-continuous-parameters');
                    var evidence_buttons = $('#node-parameters-plot-copy #evidence-buttons');
                    var evidence_button_clear = evidence_buttons.find("#clear-evidence");
                    var waiting_div = $('#node-parameters-plot-copy .waiting-param-node');
                    var node_parameters_root = $('#node-parameters-plot-root #node-parameters-0');
                    var placeholder_node_plot = $("#node-parameters-plot-copy #node-html-plot");
                    var node_name_plot = $("#node-parameters-plot-copy #node-name-plot");
                    var evidence_value = data["evidence_value"];
                    waiting_div.hide();
                    node_name_plot.text(nodeId);

                    if (evidence_value != null) {
                        evidence_button_clear.prop('disabled', false);
                    }

                    if (data_type === "discrete") {
                        var node_parameters_states = data["node_parameters_states"];
                        var node_parameters_values = data["node_parameters_values"];
                        createParametersStates(node_parameters_states, node_parameters_values, node_parameters_root, node_probability_discrete_parameters, node_probability_continuous_parameters);
                        node_probability_continuous_parameters.hide();
                        node_probability_discrete_parameters.removeClass("hide-element").show();
                    } else {
                        node_probability_continuous_parameters.removeClass("hide-element").show();

                        var parameters_plot = data["node_parameters_plot_html"];
                        var node_parameters_selection = $("#node-parameters-plot-copy #node-parameters-selection");
                        render_continuous_params(placeholder_node_plot, parameters_plot, node_parameters_selection, evidence_value);
                    }

                    parameters_height = (bayesian_network_graph.height() / 2) + node_data_captor.y - (placeholder_node_params_copy.height() + 60);
                    placeholder_node_params_copy.css({top: parameters_height});
                    evidence_buttons.removeClass("hide-element").show();

                    evidence_buttons.find("#set-evidence").click(function (e) {
                        show_waiting_inference(node_probability_continuous_parameters, node_probability_discrete_parameters, evidence_buttons, waiting_div);
                        var new_evidence_value = node_parameters_selection.val();
                        set_nodes_evidences([nodeId], new_evidence_value, node_data_captor, false);
                    });

                    evidence_buttons.find("#clear-evidence").click(function (e) {
                        show_waiting_inference(node_probability_continuous_parameters, node_probability_discrete_parameters, evidence_buttons, waiting_div);
                        clear_nodes_evidences([nodeId], node_data_captor, false);
                    });

                }
            },
            error: function () {
                placeholder_node_params_copy.remove();
                var notify_error = print_notify_message(permanent = false, type = "danger",
                    icon = 'fa fa-align-left icon-margin-right', title = "Parameters (" + nodeId + "): ",
                    message = "Exception: No parameters or render available for this node.", delay = 150,
                    additional_message = "", position = "bottom-right");
            }
        });
    }

    function createParametersStates(node_parameters_states, node_parameters_values, node_parameters_root, node_probability_discrete_parameters, node_probability_continuous_parameters) {
        node_probability_discrete_parameters.empty();
        for (let i = 0; i < node_parameters_states.length; i++) {
            var name = node_parameters_states[i];
            var value = node_parameters_values[i];
            let cloneDiv = node_parameters_root.clone();
            cloneDiv.attr("id", "node-parameters-" + i);
            cloneDiv.find('.progress-percentage-params').text((value * 100).toFixed(2) + "%");
            cloneDiv.find('#checkbox-node-parameters-0').attr("id", "checkbox-node-parameters-" + i);
            var percent = cloneDiv.find('.progress-fill-params span').html();
            cloneDiv.find('.progress-fill-params').css('width', percent);
            cloneDiv.find('.node-param-state-name').text(name);
            cloneDiv.show();
            node_probability_discrete_parameters.append(cloneDiv);

            cloneDiv.find("#checkbox-node-parameters-" + i).on("change", function () {
                var checkbox_node_parameters = $(this).is(":checked");
                if (checkbox_node_parameters) {
                    node_probability_discrete_parameters.find('.progress-fill-params span').each(function () {
                        $(this).text("0.00%");
                        $(this).parent().css('width', "0.00%");
                    });
                    node_probability_discrete_parameters.find('.node-parameters-discrete-checkbox').each(function () {
                        $(this).prop('checked', false);
                    });
                    cloneDiv.find('.node-parameters-discrete-checkbox').prop('checked', true);
                    cloneDiv.find('.progress-fill-params').css('width', "100.00%");
                    cloneDiv.find('.progress-percentage-params').text("100.00%");
                } else {
                    createParametersStates(node_parameters_states, node_parameters_values, node_parameters_root, node_probability_discrete_parameters, node_probability_continuous_parameters);
                }
            });
        }
    }

    function set_nodes_evidences(nodes_ids, evidence_value, node_data_captor, multiple_nodes_set, callback) {
        var data_client_json = {
            "nodes_ids": nodes_ids,
            "evidence_value": evidence_value,
            "evidence_scale": "scalar",
            "structure_id": $("#selectize-bn-structure").val(),
            "model_name": $("#model_name").val()
        };
        if (multiple_nodes_set) {
            data_client_json["evidence_scale"] = "num_std_deviations";
        }
        data_client_json = JSON.stringify(data_client_json);
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_set_nodes_evidences/",
            data: data_send,
            dataType: 'json',
            beforeSend: function () {
            },
            success: function (data) {
                if (data["error"]) {
                    var notify_error = print_notify_message(permanent = false, type = "danger",
                        icon = 'fa fa-align-left icon-margin-right', title = "Evidence (" + nodeId + "): ",
                        message = "Exception: error setting the evidence for this node.", delay = 150,
                        additional_message = "", position = "bottom-right");
                } else {
                    for (let i = 0; i < nodes_ids.length; i++) {
                        var node_sigmajs = sigmajs_bn.graph.getNode(nodes_ids[i]);
                        change_node_edge_color(node_sigmajs, sigmajs_bn_settings["evidence_set_color"], false);
                        node_sigmajs.selected_important = true;
                        node_sigmajs.evidence_set = true;
                    }
                    if (!multiple_nodes_set) {
                        get_and_render_node_parameters(sigmajs_bn, nodes_ids[0], node_data_captor);
                    }
                    sigmajs_bn.refresh();
                }
                show_probabilities_effect();
            },
            error: function () {
                var notify_error = print_notify_message(permanent = false, type = "danger",
                    icon = 'fa fa-align-left icon-margin-right', title = "Evidence (" + nodes_ids + "): ",
                    message = "Exception: error setting the evidence for this node.", delay = 150,
                    additional_message = "", position = "bottom-right");
            },
            complete: function () {
                if (callback !== undefined) {
                    callback();
                }
            }
        });
    }

    function show_probabilities_effect() {
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_show_probabilities_effect/",
            data: data_send,
            dataType: 'json',
            beforeSend: function () {
            },
            success: function (data) {
                var nodes_graph_evidence_effect = $("#select-nodes-graph-evidence-effect");
                var group_graph_evidence_effect_select = $("#select-group-graph-evidence-effect");
                var select_node_input_kl_divergence = $("#select-node-bn-kl-divergence");
                var select_node_input_mean_alteration = $("#select-node-bn-mean-alteration");
                var select_node_input_std_deviation_alteration = $("#select-node-bn-std-deviation-alteration");
                var kl_divergences = data["kl_divergences"];
                var mean_alteration = data["mean_alteration"];
                var std_deviation_alteration = data["std_deviation_alteration"];
                var graph_groups = data["graph_groups"];
                nodes_graph_evidence_effect.removeClass("hide-element");
                nodes_graph_evidence_effect.show();
                if (Object.keys(graph_groups).length > 0) {
                    group_graph_evidence_effect_select.removeClass("hide-element");
                    group_graph_evidence_effect_select.show();
                    var group_graph_evidence_effect = $("#select-group-graph-evidence-effect-bn");
                    group_graph_evidence_effect.selectize()[0].selectize.destroy();
                    group_graph_evidence_effect.empty();
                    var items = [];
                    for (var dict_key in graph_groups) {
                        items.push({item: dict_key});
                    }
                    items.push({"item": "No group"});
                    group_graph_evidence_effect.selectize({
                        maxItems: 1,
                        maxOptions: items.length + 1,
                        labelField: "item",
                        valueField: "item",
                        searchField: "item",
                        options: items,
                    });
                    group_graph_evidence_effect.on("change", function (e) {
                        var group_chosen_id = $(this).val();
                        if (group_chosen_id == "No group") {
                            $("#button-set-evidence-group").addClass("hide-element");
                            $("#button-clear-evidence-group").addClass("hide-element");
                        }
                        get_and_render_all_nodes_by_group(sigmajs_bn, group_chosen_id);
                        get_category_values_in_group_probabilities_effect(group_chosen_id);
                    });
                }
                create_probabilities_effect_drop_down(select_node_input_kl_divergence, kl_divergences);
                create_probabilities_effect_drop_down(select_node_input_mean_alteration, mean_alteration);
                create_probabilities_effect_drop_down(select_node_input_std_deviation_alteration, std_deviation_alteration);
            }
        });
    }

    function get_category_values_in_group_probabilities_effect(group_chosen_id) {
        var notify_loading = print_notify_message(permanent = true, type = "info",
            icon = 'fa fa-spinner fa-spin icon-margin-right', title = "Calculating groups inference effect",
            message = "Running multivariate KL-divergence", delay = 0,
            additional_message = "", position = "bottom-right");

        if (group_chosen_id == "No group") {
            reset_nodes_edges_color(sigmajs_bn);
            $("#select-category-graph-evidence-effect").hide();
        } else {
            var data_client_json = {
                "group_chosen_id": group_chosen_id,
                "model_name": $("#model_name").val()
            };
            data_client_json = JSON.stringify(data_client_json);
            var data_send = {
                "csrfmiddlewaretoken": csrf_token,
                "data_client_json": data_client_json
            };
            $.ajax({
                type: "POST",
                url: "/morpho/ml_bn_show_probabilities_effect_group/",
                data: data_send,
                dataType: 'json',
                beforeSend: function () {
                },
                success: function (data) {
                    $("#select-category-graph-evidence-effect").removeClass("hide-element");
                    $("#select-category-graph-evidence-effect").show();
                    if (data["kl_divergences_group_categories"].length > 0) {
                        var select_category_input = $("#select-category-evidence-effect-bn");
                        select_category_input.selectize()[0].selectize.destroy();
                        select_category_input.empty();
                        var items = data["kl_divergences_group_categories"].map(function (x) {
                            return {item: x[0] + " " + x[1]};
                        });
                        items.push({"item": "All categories"});
                        select_category_input.selectize({
                            maxItems: 1,
                            maxOptions: data["kl_divergences_group_categories"].length + 1,
                            labelField: "item",
                            valueField: "item",
                            searchField: "item",
                            options: items,
                        });
                        select_category_input.on("change", function (e) {
                            var category_chosen_id = $(this).val();
                            if (category_chosen_id == "All categories") {
                                reset_nodes_edges_color(sigmajs_bn);
                                sigmajs_bn_settings["active_category"] = "";
                                get_and_render_all_nodes_by_group(sigmajs_bn, sigmajs_bn_settings["active_group"]);
                            } else {
                                category_chosen_id = category_chosen_id.split(' ')[0];
                                get_all_nodes_in_category(sigmajs_bn, category_chosen_id, "render_nodes_in_category");
                            }
                        });
                    } else {
                        $("#select-category-graph-evidence-effect").hide();
                    }
                },
                complete: function () {
                    notify_loading.close();
                }
            });
        }
    }


    function focus_single_node(node_chosen_id) {
        var node_sigmajs = sigmajs_bn.graph.getNode(node_chosen_id);

        sigmajs_bn.renderers[0].dispatchEvent('clickNode', node_sigmajs);
    }

    function create_probabilities_effect_drop_down(select_element, nodes) {
        var items = nodes.map(function (x) {
            return {item: x[0] + ' ' + x[1].toFixed(2)};
        });

        select_element.selectize()[0].selectize.destroy();
        select_element.empty();
        select_element.selectize({
            maxItems: 1,
            maxOptions: nodes.length,
            labelField: "item",
            valueField: "item",
            searchField: "item",
            options: items,
        });
        select_element.on("change", function (e) {
            var node_chosen_id = $(this).val();
            if (node_chosen_id !== "" && node_chosen_id !== undefined) {
                node_chosen_id = node_chosen_id.split(" ")[0];
                focusOnNodeSigmajs(sigmajs_bn, node_chosen_id);
                focus_single_node(node_chosen_id);
            }
        });
    }

    function clear_nodes_evidences(nodes_ids, node_data_captor, multiple_nodes_set) {
        var data_client_json = {
            "nodes_ids": nodes_ids,
            "structure_id": $('#selectize-bn-structure').val(),
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json);
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_clear_nodes_evidences/",
            data: data_send,
            dataType: 'json',
            beforeSend: function () {
            },
            success: function (data) {
                if (data["error"]) {
                    var notify_error = print_notify_message(permanent = false, type = "danger",
                        icon = 'fa fa-align-left icon-margin-right', title = "Evidence (" + nodes_ids + "): ",
                        message = "Exception: error clearing the evidence for this node.", delay = 150,
                        additional_message = "", position = "bottom-right");
                } else {
                    for (let i = 0; i < nodes_ids.length; i++) {
                        var node_sigmajs = sigmajs_bn.graph.getNode(nodes_ids[i]);
                        if (node_sigmajs.color_important !== undefined) {
                            change_node_edge_color(node_sigmajs, node_sigmajs.color_important, false);
                        } else {
                            change_node_edge_color(node_sigmajs, node_sigmajs.originalColor, false);
                            node_sigmajs.selected_important = false;
                        }
                        node_sigmajs.evidence_set = false;
                    }

                    if (!multiple_nodes_set) {
                        get_and_render_node_parameters(sigmajs_bn, nodes_ids[0], node_data_captor);
                    }
                    var nodes = sigmajs_bn.graph.nodes();
                    var i = 0;
                    var is_evidence_set_graph = false;
                    while (!is_evidence_set_graph && i < nodes.length) {
                        if (nodes[i].evidence_set === true) {
                            is_evidence_set_graph = true;
                        }
                        i = i + 1;
                    }
                    if (!is_evidence_set_graph) {
                        $("#select-nodes-graph-evidence-effect").hide();
                        $("#select-group-graph-evidence-effect").hide();
                        $("#select-category-graph-evidence-effect").hide();
                    }

                    sigmajs_bn.refresh();
                }
            },
            error: function () {
                var notify_error = print_notify_message(permanent = false, type = "danger",
                    icon = 'fa fa-align-left icon-margin-right', title = "Evidence (" + nodes_ids + "): ",
                    message = "Exception: error clearing the evidence for this node.", delay = 150,
                    additional_message = "", position = "bottom-right");
            }
        });
    }

    function get_and_render_node_connections_info(sigmajs_instance, nodeId, highlight_node_color) {
        var data_client_json = {
            "node_id": nodeId,
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json);
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_get_node_connections_info/",
            data: data_send,
            dataType: 'json',
            success: function (data) {
                if (data["error"]) {
                    var message_error = "Fatal error" + data["error_message"];
                    var notify_error = print_notify_message(permanent = false, type = "danger",
                        icon = 'fa fa-link icon-margin-right', title = "Error",
                        message = message_error, delay = 150,
                        additional_message = "", position = "bottom-right");
                } else {
                    if ($('[data-notify="container"]').length > 2) {
                        $.notifyClose();
                    }
                    set_node_color(sigmajs_instance, nodeId, highlight_node_color);

                    var top_parents = "Top parents (" + data["top_parents"].length + "): " + list_tuple_to_str(data["top_parents"]);
                    var top_chidlren = "Top children (" + data["top_children"].length + "): " + list_tuple_to_str(data["top_children"]);

                    var message = "Num parents: " + data["num_parents"] + "<br>" + "Num children: " + data["num_children"]
                        + "<br>" + "Num neighbors: " + data["num_neighbors"] + "<br>" + top_parents + "<br>" + top_chidlren;


                    var notify_parameters = print_notify_message(permanent = true, type = "info",
                        icon = 'fa fa-link icon-margin-right', title = "Connections (" + nodeId + "): ",
                        message = message, delay = 4000,
                        additional_message = "", position = "bottom-right");
                }
            },
            error: function () {
                var notify_error = print_notify_message(permanent = false, type = "danger",
                    icon = 'fa fa-align-left icon-margin-right', title = "Parameters (" + nodeId + "): ",
                    message = "Exception: No parameters or render available for this node.", delay = 150,
                    additional_message = "", position = "bottom-right");
            }
        });
    }

    function get_and_render_edge_info(sigmajs_instance, source_node, target_node) {
        var data_client_json = {
            "source_node": source_node,
            "target_node": target_node,
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json)
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };

        var message_error = "No info available for this edge.";

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_get_edge_info/",
            data: data_send,
            dataType: 'json',
            success: function (data) {
                if (data["error"]) {
                    var notify_error = print_notify_message(permanent = false, type = "danger",
                        icon = 'fa fa-info icon-margin-right', title = "Error",
                        message = message_error, delay = 150,
                        additional_message = "", position = "bottom-right");
                } else {
                    if ($('[data-notify="container"]').length > 4) {
                        $.notifyClose();
                    }
                    var message = "Weight: " + data["weight"]
                    var notify_parameters = print_notify_message(permanent = true, type = "info",
                        icon = 'fa fa-info icon-margin-right', title = "Edge (" + source_node + "-" + target_node + "): ",
                        message = message, delay = 4000,
                        additional_message = "", position = "bottom-right");
                }
            },
            error: function () {
                var notify_error = print_notify_message(permanent = false, type = "danger",
                    icon = 'fa fa-info icon-margin-right', title = "Edge (" + source_node + "-" + target_node + "): ",
                    message = message_error, delay = 150,
                    additional_message = "", position = "bottom-right");
            }
        });
    }

    function get_important_nodes_degrees(highlight_algorithm_selected, sigmajs_default_settings, num_neighbors, include_neighbors, propagate_colors_to_hubs_neighbors) {
        var notify_loading = print_notify_message(permanent = true, type = "info",
            icon = 'fa fa-spinner fa-spin icon-margin-right', title = "Loading algorithm",
            message = "Loading algorithm", delay = 0,
            additional_message = "", position = "bottom-right");
        var data_client_json = {
            "selection_option": highlight_algorithm_selected,
            "num_neighbors": num_neighbors,
            "include_neighbors": include_neighbors,
            "max_node_size": sigmajs_default_settings["maxNodeSize"],
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json);
        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json,
        };
        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_highlight_important_nodes/",
            data: data_send,
            dataType: 'json',
            success: function (data) {
                if (!$.isEmptyObject(data["important_nodes"])) {
                    for (var node_id in data["important_nodes"]) {
                        var params = data["important_nodes"][node_id];
                        var node_sigmajs = sigmajs_bn.graph.getNode(node_id);
                        var color_hub = params["color"];
                        if (!node_sigmajs.selected_important) {
                            change_node_edge_color(node_sigmajs, params["color"], true);
                        } else {
                            color_hub = node_sigmajs.color;
                        }
                        node_sigmajs.size = params["size"];
                        sigmajs_bn.settings('maxNodeSize', params["size"]);
                        node_sigmajs.selected_important = true;

                        if (include_neighbors) {
                            var neighbors = params["neighbors"];
                            neighbors.forEach(function (neighbor_node_id) {
                                var neighbor_node_sigmajs = sigmajs_bn.graph.getNode(neighbor_node_id);
                                if (propagate_colors_to_hubs_neighbors && neighbor_node_sigmajs.selected_important) {
                                    var hubs_of_neighbor = [];

                                    for (var node_id_ in data["important_nodes"]) {
                                        var params_ = data["important_nodes"][node_id_];
                                        var neighbors_ = params_["neighbors"];
                                        neighbors_.forEach(function (neighbor_node_id_) {
                                            if (neighbor_node_id_ === neighbor_node_id) {
                                                hubs_of_neighbor.push(node_id_);
                                                var hub_node_sigmajs = sigmajs_bn.graph.getNode(node_id_);
                                                change_node_edge_color(hub_node_sigmajs, color_hub, true);
                                            }
                                        });
                                    }

                                    for (var hub_node_idx in hubs_of_neighbor) {
                                        var hub_node_id = hubs_of_neighbor[hub_node_idx];
                                        var hub_params = data["important_nodes"][hub_node_id]
                                        var neighbors_of_hub = hub_params["neighbors"];

                                        neighbors_of_hub.forEach(function (neighbor_hub_id) {
                                            var neighbor_hub = sigmajs_bn.graph.getNode(neighbor_hub_id);
                                            change_node_edge_color(neighbor_hub, color_hub, false);
                                        });
                                    }
                                }
                                change_node_edge_color(neighbor_node_sigmajs, color_hub, false);
                                neighbor_node_sigmajs.selected_important = true;
                            });
                        }
                    }
                    if (include_neighbors) {
                        for (var node_id in data["important_nodes"]) {
                            var params = data["important_nodes"][node_id];
                            var node_sigmajs = sigmajs_bn.graph.getNode(node_id);
                            var color_hub = node_sigmajs.color;
                            var neighbors = params["neighbors"];
                            neighbors.forEach(function (neighbor_node_id) {
                                var neighbor_node_sigmajs = sigmajs_bn.graph.getNode(neighbor_node_id);
                                change_node_edge_color(neighbor_node_sigmajs, color_hub, true);
                            });
                        }
                    }
                    sigmajs_bn_settings["nodes_slider"].noUiSlider.updateOptions({
                        start: [sigmajs_bn.settings('maxNodeSize')],
                        range: {
                            'min': sigmajs_bn.settings('minNodeSize'),
                            'max': parseFloat(sigmajs_bn.settings('maxNodeSize').toFixed(1))
                        }
                    });
                    sigmajs_bn.refresh();
                }
            },
            error: function () {
            },
            complete: function () {
                notify_loading.close();
            }
        });
    }

    function get_important_nodes_betweenness(highlight_algorithm_selected, sigmajs_default_settings) {
        var notify_loading = print_notify_message(permanent = true, type = "info",
            icon = 'fa fa-spinner fa-spin icon-margin-right', title = "Loading algorithm",
            message = "Loading algorithm", delay = 0,
            additional_message = "", position = "bottom-right");
        var data_client_json = {
            "selection_option": highlight_algorithm_selected,
            "max_node_size": sigmajs_default_settings["maxNodeSize"],
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json);

        var data_send = {
            "csrfmiddlewaretoken": csrf_token,
            "data_client_json": data_client_json
        };
        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_highlight_important_nodes/",
            data: data_send,
            dataType: 'json',
            success: function (data) {
                if (!$.isEmptyObject(data["important_nodes"])) {
                    var maxNodeSize = sigmajs_bn.settings('maxNodeSize');
                    for (var node_id in data["important_nodes"]) {
                        var node = data["important_nodes"][node_id];
                        var node_sigmajs = sigmajs_bn.graph.getNode(node_id);

                        if (node.size > 0) {
                            node_sigmajs.size = node.size;
                            change_node_edge_color(node_sigmajs, node.color, true);
                            node_sigmajs.selected_important = true;
                            if (node_sigmajs.size > maxNodeSize) {
                                maxNodeSize = node_sigmajs.size;
                            }
                        }
                    }
                    if (maxNodeSize > sigmajs_bn.settings('maxNodeSize')) {
                        sigmajs_bn.settings('maxNodeSize', maxNodeSize);
                    }
                    sigmajs_bn_settings["nodes_slider"].noUiSlider.updateOptions({
                        start: [sigmajs_bn.settings('maxNodeSize')],
                        range: {
                            'min': sigmajs_bn.settings('minNodeSize'),
                            'max': parseFloat(sigmajs_bn.settings('maxNodeSize').toFixed(1))
                        }
                    });
                    sigmajs_bn.refresh();
                }
            }, error: function () {
            },
            complete: function () {
                notify_loading.close();
            }
        });
    }

    function get_num_common_structures(edge, structures_multi_selected, break_if_multi) {
        var num_common_structures = 0;
        for (i = 0; i <= edge.structure_id.length; i++) {
            if (structures_multi_selected.indexOf(edge.structure_id[i]) !== -1) {
                num_common_structures++;
                if (num_common_structures > 1) {
                    if (break_if_multi) {
                        break;
                    }
                }
            }
        }
        return num_common_structures;
    }

    function color_common_structures_edges(edge, structures_multi_selected) {
        var num_common_structures = get_num_common_structures(edge, structures_multi_selected, true);
        if (num_common_structures > 1) {
            change_node_edge_color(edge, sigmajs_bn_settings["color_common_edges"], false);
        } else {
            hide_node_edge(edge);
        }
    }

    function reset_nodes_edges_color_prob_clustering() {
        var structure_id = $('#selectize-bn-structure').val();

        sigmajs_bn_settings["active_structure"] = structure_id;

        // Select the nodes (and their neighbors with edges in this cluster:
        if (sigmajs_bn_settings["active_category"] !== "") {
            var notify_loading = undefined;
            get_all_nodes_in_category(sigmajs_bn, sigmajs_bn_settings["active_category"],
                "render_nodes_in_category", notify_loading);
        } else {
            reset_edges_color_prob_clustering();
        }

    }


    function reset_edges_color_prob_clustering() {
        var structure_id = $('#selectize-bn-structure').val();
        var structure_common_edges = $('#selectize-bn-structure-common-edges').val();
        var structures_multi_selected = $('#selectize-multi-structures').val();

        if (structure_id === "all" && (structure_common_edges !== "" && structure_common_edges !== "no-common")) {
            structure_id = structure_common_edges;
        }

        if (structures_multi_selected.length === 0) {
            structures_multi_selected = Object.keys(sigmajs_bn_settings["structures_info"])
        } else if (structure_id === "all" && structures_multi_selected.length === 1) {
            structure_id = structures_multi_selected[0];
        }

        var num_structures = structures_multi_selected.length;
        var num_edges_selected = 0;

        sigmajs_bn.graph.edges().forEach(function (edge) {
            edge.structure_selected = true;

            if ((sigmajs_bn_settings["active_category"] !== "" && edge.selected) || sigmajs_bn_settings["active_category"] === "") {
                if (structure_id === "all") {
                    if (edge.structure_id.length == 1) {
                        if (structures_multi_selected.indexOf(edge.structure_id[0]) !== -1) {
                            change_node_edge_color(edge, sigmajs_bn_settings["structures_info"][edge.structure_id[0]]["color"], false);
                        } else {
                            hide_node_edge(edge);
                        }
                    } else {
                        color_common_structures_edges(edge, structures_multi_selected);
                    }
                } else if (structure_id === "common") {
                    color_common_structures_edges(edge, structures_multi_selected);
                } else if (structure_id === "common-all") {
                    var num_common_structures = get_num_common_structures(edge, structures_multi_selected, false);
                    if (num_common_structures === num_structures) {
                        change_node_edge_color(edge, sigmajs_bn_settings["color_common_edges"], false);
                    } else {
                        hide_node_edge(edge);
                    }
                } else { //one specific structure
                    if (edge.structure_id.indexOf(structure_id) !== -1) {
                        change_node_edge_color(edge, sigmajs_bn_settings["structures_info"][structure_id]["color"], false);
                    } else {
                        hide_node_edge(edge);
                    }
                }
            }

            var hide_edges = $("#checkbox-on-click-group-hide-others").is(":checked");
            if (hide_edges) {
                if (!edge.hidden) {
                    num_edges_selected++;
                }
                edge.originalColor = edge.color;
            } else {
                if (edge.color !== sigmajs_bn_settings["color_hidden"]) {
                    num_edges_selected++;
                    edge.originalColor = edge.color;
                }
            }

            if (sigmajs_bn_settings["active_category"] === "" && edge.selected) {
                edge.selected = false;
            }
        });

        $("#network-stats-num-edges-selected").text(num_edges_selected);

        sigmajs_bn.refresh();
    }

    $("#update-filter-edges-by-weight").click(function (e) {
        var notify_loading = print_notify_message(permanent = true, type = "info",
            icon = 'fa fa-spinner fa-spin icon-margin-right', title = "Filter edges",
            message = "Filtering edges by weights...", delay = 4000, additional_message = "");

        var weights_range = sigmajs_bn_settings["filter_edges_weight_slider"].noUiSlider.get();

        var data_client_json = {
            "weights_range": weights_range,
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json);
        var data_send = {
            "csrfmiddlewaretoken": $("#csrf_token").val(),
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_bn_filter_edges_by_weight/",
            data: data_send,
            dataType: 'json',
            success: function (data) {
                notify_loading.close();
                if (data["error"]) {
                    var notify_error = print_notify_message(permanent = false, type = "danger", icon = 'fa fa-spinner fa-spin icon-margin-right', title = "Error filtering edges", message = data["error_message"], delay = 3000);
                } else {
                    var layout_name = sigmajs_bn_settings["active_layout"];
                    var additional_params = undefined;
                    sigmajs_bn_settings["num_nodes"] = data["num_nodes"];
                    sigmajs_bn_settings["num_edges"] = data["num_edges"];
                    print_current_num_nodes_edges();
                    change_layout_sigmajs(layout_name, additional_params);
                }
            },
            error: function () {
                notify_loading.close();
                var notify_error = print_notify_message(permanent = false, type = "danger", icon = 'fa fa-spinner fa-spin icon-margin-right', title = "Error filtering edges", message = data["error_message"], delay = 3000);
            }
        });
    });

    function init_sliders_glasso() {
        var slider_glasso_alpha = $("#glasso-alpha")[0];
        var slider_glasso_tol = $("#glasso-tol")[0];
        var slider_glasso_max_iter = $("#glasso-max-iter")[0];

        var min_val_alpha = 0.0001;
        noUiSlider.create(slider_glasso_alpha, {
            start: [min_val_alpha],
            connect: true,
            format: wNumb({
                decimals: 4
            }),
            range: {
                'min': min_val_alpha,
                'max': 0.01,
            },
            tooltips: [true],
            step: 0.0001,
            pips: {
                mode: 'count',
                values: 3,
                density: 1,
                format: wNumb({
                    decimals: 2,
                }),
            }
        });

        var min_val_tol = 0.0001;
        noUiSlider.create(slider_glasso_tol, {
            start: [min_val_tol],
            connect: true,
            format: wNumb({
                decimals: 4
            }),
            range: {
                'min': min_val_tol,
                'max': 1,
            },
            tooltips: [true],
            step: 0.0001,
            pips: {
                mode: 'count',
                values: 2,
                density: 1,
                format: wNumb({
                    decimals: 3,
                }),
            }
        });

        var min_val_max_iter = 1000;
        noUiSlider.create(slider_glasso_max_iter, {
            start: [min_val_max_iter],
            connect: true,
            format: wNumb({
                decimals: 0
            }),
            range: {
                'min': min_val_max_iter,
                'max': 10000,
            },
            tooltips: [true],
            step: 1,
            pips: {
                mode: 'count',
                values: 3,
                density: 1,
                format: wNumb({
                    decimals: 0,
                }),
            }
        });

        set_input_left_for_slider("glasso-alpha", slider_glasso_alpha, min_val_alpha, function () {
        });
        set_input_left_for_slider("glasso-tol", slider_glasso_tol, min_val_tol, function () {
        });
        set_input_left_for_slider("glasso-max-iter", slider_glasso_max_iter, min_val_max_iter, function () {
        });

    }

    $("#update-params-glasso").click(update_glasso_params);

    function update_glasso_params() {
        var notify_loading = print_notify_message(permanent = true, type = "info",
            icon = 'fa fa-spinner fa-spin icon-margin-right', title = "Loading layout",
            message = "Updating graphical lasso parameters...", delay = 4000, additional_message = "");

        var alpha = $("#glasso-alpha")[0].noUiSlider.get();
        var tol = $("#glasso-tol")[0].noUiSlider.get();
        var max_iter = $("#glasso-max-iter")[0].noUiSlider.get();

        var data_client_json = {
            "alpha": alpha,
            "tol": tol,
            "max_iter": max_iter,
            "model_name": $("#model_name").val()
        };
        data_client_json = JSON.stringify(data_client_json);
        var data_send = {
            "csrfmiddlewaretoken": $("#csrf_token").val(),
            "data_client_json": data_client_json,
        };

        $.ajax({
            type: "POST",
            url: "/morpho/ml_prob_clustering_update_glasso/",
            data: data_send,
            dataType: 'json',
            success: function (data) {
                notify_loading.close();
                if (data["error"]) {
                    var notify_error = print_notify_message(permanent = false, type = "danger", icon = 'fa fa-spinner fa-spin icon-margin-right', title = "Error loading layout", message = data["error_message"], delay = 3000);
                } else {
                    var layout_name = sigmajs_bn_settings["active_layout"];
                    var additional_params = undefined;
                    sigmajs_bn_settings["num_nodes"] = data["num_nodes"];
                    sigmajs_bn_settings["num_edges"] = data["num_edges"];
                    print_current_num_nodes_edges();
                    change_layout_sigmajs(layout_name, additional_params);
                }
            },
            error: function () {
                notify_loading.close();
                var notify_error = print_notify_message(permanent = false, type = "danger", icon = 'fa fa-spinner fa-spin icon-margin-right', title = "Error updating graphical lasso parameters", message = "Server error", delay = 3000);
            }
        });
    }

    function print_current_num_nodes_edges() {
        $("#network-stats-num-nodes").text(sigmajs_bn_settings["num_nodes"]);
        $("#network-stats-num-edges").text(sigmajs_bn_settings["num_edges"]);
        $("#network-stats-num-nodes-selected").text(sigmajs_bn_settings["num_nodes"]);
        $("#network-stats-num-edges-selected").text(sigmajs_bn_settings["num_edges"]);
    }

    function update_num_nodes_edges_selected() {
        var num_nodes = 0;
        var num_edges = 0;
        sigmajs_bn.graph.edges().forEach(function (edge) {
            if (edge.selected) {
                num_edges++;
            }
        });
        sigmajs_bn.graph.nodes().forEach(function (node) {
            if (node.selected) {
                num_nodes++;
            }
        });

        $("#network-stats-num-nodes-selected").text(sigmajs_bn_settings["num_nodes"]);
        $("#network-stats-num-edges-selected").text(sigmajs_bn_settings["num_edges"]);
    }

    $('#selectize-bn-structure-sort').on("change", function (e) {
        var selectize_structure = $("#selectize-bn-structure")[0].selectize;
        var structures_info = sigmajs_bn_settings["structures_info"];
        var filter_by = $(this).val();
        initialize_selectize_structures(structures_info, filter_by);
        selectize_structure.refreshOptions();
    });

    function list_tuple_to_str(list_tuples) {
        var str_result = "[";
        for (var i = 0; i < list_tuples.length; i++) {
            var tuple = list_tuples[i];

            for (var j = 0; j < tuple.length; j++) {
                if (j == 0) {
                    str_result += tuple[j];
                } else {
                    str_result += " (" + tuple[j] + ")";
                }
            }

            if (i !== list_tuples.length - 1) {
                str_result += ", ";
            }
        }
        str_result += "]";

        return str_result;
    }

});
